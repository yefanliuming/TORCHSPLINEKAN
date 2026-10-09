// Fused kernels for the activation regularisation statistic and for the
// B-spline basis with a fused input-gradient contraction.
//
// Edge statistic (paper eq. 19), per edge (o, i):
//     phi[n,o,i] = s[n,i] * bw[o,i] + sum_r H[n,i,r] * W[o,i,r]
//     a[o,i]     = (1/N) * sum_n |phi[n,o,i]|
// where s = base_activation(x), H = basis(x), W = effective spline weight.
// phi is never materialised: the forward accumulates |phi| on the fly and the
// backward recomputes phi (R FMAs per (n,o,i)) to obtain sign(phi).

#include "kan_ops.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int EA_THREADS = 128;   // threads (output channels) per block, kernel A
constexpr int EA_TILE_N  = 32;    // samples staged in shared memory per step
constexpr int EB_THREADS = 128;   // threads (samples) per block, kernel B
constexpr int EB_TILE_O  = 32;    // output channels staged per step

template <typename T>
__device__ __forceinline__ T sgn(T v) {
    return (T)((v > (T)0) - (v < (T)0));   // sign(0) = 0, matching d|x|/dx in PyTorch
}

// Kernel A: one thread per (o, i), loops over a slice of n.
//   GRAD = false : a[o,i]   += sum_n |phi| / N
//   GRAD = true  : dbw[o,i] += sum_n g * s,  dW[o,i,r] += sum_n g * H[n,i,r]
//                  with g = ga[o,i] * sign(phi) / N
template <typename T, int RMAX, bool GRAD>
__global__ void edge_stat_kernel(
    const T* __restrict__ s, const T* __restrict__ H,
    const T* __restrict__ bw, const T* __restrict__ W,
    const T* __restrict__ ga,
    T* __restrict__ out_a, T* __restrict__ dbw, T* __restrict__ dW,
    int N, int I, int O, int R, int n_per_split, bool has_base, bool atomic
) {
    __shared__ T Hs[EA_TILE_N * RMAX];
    __shared__ T ss[EA_TILE_N];

    const int i = blockIdx.y;
    const int o = blockIdx.x * EA_THREADS + threadIdx.x;
    const bool valid = o < O;
    const int n_begin = blockIdx.z * n_per_split;
    const int n_end = min(N, n_begin + n_per_split);
    const T invN = (T)1 / (T)N;

    T w[RMAX];
    T dw[RMAX];
    T bwv = 0, gav = 0;
#pragma unroll
    for (int r = 0; r < RMAX; ++r) {
        w[r] = (valid && r < R) ? W[((int64_t)o * I + i) * R + r] : (T)0;
        dw[r] = 0;
    }
    if (valid && has_base) bwv = bw[(int64_t)o * I + i];
    if (GRAD && valid) gav = ga[(int64_t)o * I + i] * invN;
    T acc = 0, dbwv = 0;

    for (int n0 = n_begin; n0 < n_end; n0 += EA_TILE_N) {
        const int tn = min(EA_TILE_N, n_end - n0);
        for (int t = threadIdx.x; t < tn * R; t += EA_THREADS) {
            const int nn = t / R, r = t - nn * R;
            Hs[nn * RMAX + r] = H[((int64_t)(n0 + nn) * I + i) * R + r];
        }
        if (has_base) {
            for (int t = threadIdx.x; t < tn; t += EA_THREADS)
                ss[t] = s[(int64_t)(n0 + t) * I + i];
        }
        __syncthreads();
        if (valid) {
            for (int t = 0; t < tn; ++t) {
                T phi = has_base ? ss[t] * bwv : (T)0;
#pragma unroll
                for (int r = 0; r < RMAX; ++r)
                    if (r < R) phi += Hs[t * RMAX + r] * w[r];
                if (!GRAD) {
                    acc += fabs(phi);
                } else {
                    const T g = gav * sgn(phi);
                    if (has_base) dbwv += g * ss[t];
#pragma unroll
                    for (int r = 0; r < RMAX; ++r)
                        if (r < R) dw[r] += g * Hs[t * RMAX + r];
                }
            }
        }
        __syncthreads();
    }
    if (!valid) return;
    const int64_t e = (int64_t)o * I + i;
    if (!GRAD) {
        if (atomic) atomicAdd(out_a + e, acc * invN); else out_a[e] = acc * invN;
    } else {
        if (has_base && dbw != nullptr) {
            if (atomic) atomicAdd(dbw + e, dbwv); else dbw[e] = dbwv;
        }
#pragma unroll
        for (int r = 0; r < RMAX; ++r) {
            if (r < R) {
                if (atomic) atomicAdd(dW + e * R + r, dw[r]); else dW[e * R + r] = dw[r];
            }
        }
    }
}
// Kernel B: one thread per (n, i), loops over o.
//   ds[n,i]   = sum_o g[n,o,i] * bw[o,i]
//   dH[n,i,r] = sum_o g[n,o,i] * W[o,i,r],   g = ga[o,i] * sign(phi[n,o,i]) / N
template <typename T, int RMAX>
__global__ void edge_stat_input_grad_kernel(
    const T* __restrict__ s, const T* __restrict__ H,
    const T* __restrict__ bw, const T* __restrict__ W,
    const T* __restrict__ ga,
    T* __restrict__ ds, T* __restrict__ dH,
    int N, int I, int O, int R, bool has_base, bool want_ds, bool want_dH
) {
    __shared__ T Ws[EB_TILE_O * RMAX];
    __shared__ T bws[EB_TILE_O];
    __shared__ T gas[EB_TILE_O];

    const int i = blockIdx.y;
    const int n = blockIdx.x * EB_THREADS + threadIdx.x;
    const bool valid = n < N;
    const T invN = (T)1 / (T)N;

    T h[RMAX];
    T dh[RMAX];
#pragma unroll
    for (int r = 0; r < RMAX; ++r) {
        h[r] = (valid && r < R) ? H[((int64_t)n * I + i) * R + r] : (T)0;
        dh[r] = 0;
    }
    const T sv = (valid && has_base) ? s[(int64_t)n * I + i] : (T)0;
    T dsv = 0;

    for (int o0 = 0; o0 < O; o0 += EB_TILE_O) {
        const int to = min(EB_TILE_O, O - o0);
        for (int t = threadIdx.x; t < to * R; t += EB_THREADS) {
            const int oo = t / R, r = t - oo * R;
            Ws[oo * RMAX + r] = W[((int64_t)(o0 + oo) * I + i) * R + r];
        }
        for (int t = threadIdx.x; t < to; t += EB_THREADS) {
            const int64_t e = (int64_t)(o0 + t) * I + i;
            bws[t] = has_base ? bw[e] : (T)0;
            gas[t] = ga[e] * invN;
        }
        __syncthreads();
        if (valid) {
            for (int t = 0; t < to; ++t) {
                T phi = sv * bws[t];
#pragma unroll
                for (int r = 0; r < RMAX; ++r)
                    if (r < R) phi += h[r] * Ws[t * RMAX + r];
                const T g = gas[t] * sgn(phi);
                dsv += g * bws[t];
                if (want_dH) {
#pragma unroll
                    for (int r = 0; r < RMAX; ++r)
                        if (r < R) dh[r] += g * Ws[t * RMAX + r];
                }
            }
        }
        __syncthreads();
    }
    if (!valid) return;
    if (want_ds && has_base) ds[(int64_t)n * I + i] = dsv;
    if (want_dH) {
#pragma unroll
        for (int r = 0; r < RMAX; ++r)
            if (r < R) dH[((int64_t)n * I + i) * R + r] = dh[r];
    }
}

// Picks how many n-splits kernel A uses: enough blocks to fill the GPU while
// keeping each split long enough to amortise the atomic epilogue.
int edge_splits(int64_t N, int64_t blocks_xy) {
    const int64_t target_blocks = 4 * 128;   // ~4 waves of blocks on large GPUs
    int64_t splits = (target_blocks + blocks_xy - 1) / blocks_xy;
    splits = std::max<int64_t>(1, std::min<int64_t>(splits, (N + 63) / 64));
    return (int)splits;
}

template <typename T, int RMAX>
void launch_edge_stat(const torch::Tensor& s, const torch::Tensor& H, const torch::Tensor& bw,
                      const torch::Tensor& W, torch::Tensor& a, bool has_base, cudaStream_t st) {
    const int N = H.size(0), I = H.size(1), R = H.size(2), O = W.size(0);
    const int bx = (O + EA_THREADS - 1) / EA_THREADS;
    const int splits = edge_splits(N, (int64_t)bx * I);
    const int nps = (N + splits - 1) / splits;
    dim3 grid(bx, I, splits);
    edge_stat_kernel<T, RMAX, false><<<grid, EA_THREADS, 0, st>>>(
        has_base ? s.data_ptr<T>() : nullptr, H.data_ptr<T>(),
        has_base ? bw.data_ptr<T>() : nullptr, W.data_ptr<T>(), nullptr,
        a.data_ptr<T>(), nullptr, nullptr, N, I, O, R, nps, has_base, splits > 1);
}

template <typename T, int RMAX>
void launch_edge_stat_backward(const torch::Tensor& s, const torch::Tensor& H, const torch::Tensor& bw,
                               const torch::Tensor& W, const torch::Tensor& ga,
                               torch::Tensor& dbw, torch::Tensor& dW, torch::Tensor& ds, torch::Tensor& dH,
                               bool has_base, bool want_param, bool want_ds, bool want_dH,
                               cudaStream_t st) {
    const int N = H.size(0), I = H.size(1), R = H.size(2), O = W.size(0);
    if (want_param) {
        const int bx = (O + EA_THREADS - 1) / EA_THREADS;
        const int splits = edge_splits(N, (int64_t)bx * I);
        const int nps = (N + splits - 1) / splits;
        dim3 grid(bx, I, splits);
        edge_stat_kernel<T, RMAX, true><<<grid, EA_THREADS, 0, st>>>(
            has_base ? s.data_ptr<T>() : nullptr, H.data_ptr<T>(),
            has_base ? bw.data_ptr<T>() : nullptr, W.data_ptr<T>(), ga.data_ptr<T>(),
            nullptr, has_base ? dbw.data_ptr<T>() : nullptr, dW.data_ptr<T>(),
            N, I, O, R, nps, has_base, splits > 1);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (want_ds || want_dH) {
        dim3 grid((N + EB_THREADS - 1) / EB_THREADS, I);
        edge_stat_input_grad_kernel<T, RMAX><<<grid, EB_THREADS, 0, st>>>(
            has_base ? s.data_ptr<T>() : nullptr, H.data_ptr<T>(),
            has_base ? bw.data_ptr<T>() : nullptr, W.data_ptr<T>(), ga.data_ptr<T>(),
            (want_ds && has_base) ? ds.data_ptr<T>() : nullptr,
            want_dH ? dH.data_ptr<T>() : nullptr,
            N, I, O, R, has_base, want_ds, want_dH);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

#define KAN_DISPATCH_RMAX(R, ...)                                   \
    [&] {                                                           \
        if ((R) <= 8)       { constexpr int RMAX = 8;  __VA_ARGS__(); } \
        else if ((R) <= 16) { constexpr int RMAX = 16; __VA_ARGS__(); } \
        else if ((R) <= 32) { constexpr int RMAX = 32; __VA_ARGS__(); } \
        else                { constexpr int RMAX = 72; __VA_ARGS__(); } \
    }()

void check_edge_inputs(const torch::Tensor& s, const torch::Tensor& H,
                       const torch::Tensor& bw, const torch::Tensor& W, bool has_base) {
    TORCH_CHECK(H.is_cuda() && W.is_cuda(), "edge_abs_mean: H and W must be CUDA tensors");
    TORCH_CHECK(H.dim() == 3 && W.dim() == 3, "edge_abs_mean: H must be [N,I,R], W [O,I,R]");
    TORCH_CHECK(H.size(1) == W.size(1) && H.size(2) == W.size(2), "edge_abs_mean: H/W shape mismatch");
    TORCH_CHECK(H.size(2) <= 72, "edge_abs_mean: R must be <= 72, got ", H.size(2));
    TORCH_CHECK(H.size(1) <= KAN_CUDA_MAX_IN_FEATURES, "edge_abs_mean: too many input features");
    TORCH_CHECK(H.size(0) >= 1, "edge_abs_mean: need at least one sample");
    TORCH_CHECK(H.scalar_type() == W.scalar_type(), "edge_abs_mean: dtype mismatch");
    TORCH_CHECK(H.scalar_type() == torch::kFloat32 || H.scalar_type() == torch::kFloat64,
                "edge_abs_mean: only FP32/FP64 supported");
    if (has_base) {
        TORCH_CHECK(s.sizes() == H.sizes().slice(0, 2), "edge_abs_mean: s must be [N, I]");
        TORCH_CHECK(bw.sizes() == W.sizes().slice(0, 2), "edge_abs_mean: base_weight must be [O, I]");
        TORCH_CHECK(s.scalar_type() == H.scalar_type() && bw.scalar_type() == H.scalar_type(),
                    "edge_abs_mean: dtype mismatch");
    }
}

}  // namespace

// a[o,i] = mean_n |s[n,i]*bw[o,i] + sum_r H[n,i,r]*W[o,i,r]|
// s / bw may be empty (no base branch).
torch::Tensor edge_abs_mean_cuda(torch::Tensor s, torch::Tensor H,
                                 torch::Tensor bw, torch::Tensor W) {
    const bool has_base = kan_has_base(bw);
    check_edge_inputs(s, H, bw, W, has_base);
    c10::cuda::CUDAGuard guard(H.device());
    auto st = at::cuda::getCurrentCUDAStream().stream();
    H = H.contiguous(); W = W.contiguous();
    if (has_base) { s = s.contiguous(); bw = bw.contiguous(); }
    auto a = torch::zeros({W.size(0), W.size(1)}, W.options());
    const int64_t R = H.size(2);
    AT_DISPATCH_FLOATING_TYPES(H.scalar_type(), "edge_abs_mean_cuda", [&] {
        KAN_DISPATCH_RMAX(R, [&] {
            launch_edge_stat<scalar_t, RMAX>(s, H, bw, W, a, has_base, st);
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return a;
}

// Returns (ds [N,I] | undefined, dH [N,I,R] | undefined,
//          dbw [O,I] | undefined, dW [O,I,R] | undefined)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
edge_abs_mean_backward_cuda(torch::Tensor grad_a, torch::Tensor s, torch::Tensor H,
                            torch::Tensor bw, torch::Tensor W,
                            bool need_s, bool need_H, bool need_bw, bool need_W) {
    const bool has_base = kan_has_base(bw);
    check_edge_inputs(s, H, bw, W, has_base);
    TORCH_CHECK(grad_a.sizes() == W.sizes().slice(0, 2), "edge_abs_mean_backward: grad shape");
    c10::cuda::CUDAGuard guard(H.device());
    auto st = at::cuda::getCurrentCUDAStream().stream();
    H = H.contiguous(); W = W.contiguous();
    grad_a = grad_a.contiguous().to(W.scalar_type());
    if (has_base) { s = s.contiguous(); bw = bw.contiguous(); }
    const int64_t N = H.size(0), I = H.size(1), R = H.size(2), O = W.size(0);
    const bool want_param = need_W || (need_bw && has_base);
    const bool want_ds = need_s && has_base;
    // Kernel A writes dW unconditionally when it runs; allocate it if either is needed.
    torch::Tensor dW = want_param ? torch::zeros({O, I, R}, W.options()) : torch::Tensor();
    torch::Tensor dbw = (want_param && has_base) ? torch::zeros({O, I}, W.options()) : torch::Tensor();
    torch::Tensor ds = want_ds ? torch::empty({N, I}, H.options()) : torch::Tensor();
    torch::Tensor dH = need_H ? torch::empty({N, I, R}, H.options()) : torch::Tensor();
    AT_DISPATCH_FLOATING_TYPES(H.scalar_type(), "edge_abs_mean_backward_cuda", [&] {
        KAN_DISPATCH_RMAX(R, [&] {
            launch_edge_stat_backward<scalar_t, RMAX>(
                s, H, bw, W, grad_a, dbw, dW, ds, dH,
                has_base, want_param, want_ds, need_H, st);
        });
    });
    return {ds, dH, need_bw ? dbw : torch::Tensor(), need_W ? dW : torch::Tensor()};
}
