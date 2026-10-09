#include "kan_ops.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CUDA kernels: B-spline basis evaluation
// Each thread handles one (b, i) pair.
// grid dims: (ceil(B/32), I), block dims: (32, 1)
// Shared memory holds the knot vector for input channel i.
// ---------------------------------------------------------------------------

template <int K, typename scalar_t>
__global__ void basis_eval_kernel(
    const scalar_t* __restrict__ x,      // [B, I]
    const scalar_t* __restrict__ grid,   // [I, G+2K+1]
    scalar_t* __restrict__ H,            // [B, I, R]  R = G+K
    int B, int I, int G
) {
    // Shared memory: one knot row per block, size N = G+2K+1
    extern __shared__ char smem_raw[];
    scalar_t* sknots = reinterpret_cast<scalar_t*>(smem_raw);

    const int i = blockIdx.y;           // input channel index
    const int b = blockIdx.x * 32 + threadIdx.x; // batch index

    const int N = G + 2 * K + 1;       // total knots
    const int R = G + K;               // basis functions

    // Cooperatively load knot row into shared memory
    for (int t = threadIdx.x; t < N; t += 32) {
        sknots[t] = grid[i * N + t];
    }
    __syncthreads();

    if (b >= B) return;

    scalar_t xval = x[b * I + i];

    // Stack for Cox-de Boor: we need up to N-1 values at degree 0,
    // then reduce each iteration. Use a fixed-size local array.
    // Max N = 64+2*3+1 = 71; but we only need N-1 = 70 degree-0 values.
    // To stay register-friendly, declare a fixed array sized for K<=3.
    // degree-0: d0[r] for r=0..N-2
    // After K iterations: R values remain.
    // We store two alternating buffers.
    // Bound enforced on the host by kan_check_inputs (G + 2K <= 71).
    scalar_t buf0[KAN_CUDA_MAX_DEG0_BASIS];
    scalar_t buf1[KAN_CUDA_MAX_DEG0_BASIS];

    // Degree 0
    int n0 = N - 1;
    for (int r = 0; r < n0; ++r) {
        scalar_t lo = sknots[r];
        scalar_t hi = sknots[r + 1];
        bool inside = (xval >= lo) && (xval < hi);
        // Special case: last interval includes right endpoint
        if (r == n0 - 1) {
            inside = inside || (xval == hi);
        }
        buf0[r] = inside ? (scalar_t)1.0 : (scalar_t)0.0;
    }

    scalar_t* cur = buf0;
    scalar_t* nxt = buf1;

    // Degree d recurrence
    for (int d = 1; d <= K; ++d) {
        int n_cur = n0 - (d - 1); // entries in cur
        int n_nxt = n_cur - 1;
        for (int r = 0; r < n_nxt; ++r) {
            scalar_t t_r   = sknots[r];
            scalar_t t_rd  = sknots[r + d];
            scalar_t t_r1  = sknots[r + 1];
            scalar_t t_rd1 = sknots[r + d + 1];

            scalar_t dL = t_rd  - t_r;
            scalar_t dR = t_rd1 - t_r1;

            scalar_t left  = (dL != (scalar_t)0.0) ?
                              cur[r]   * (xval - t_r)  / dL : (scalar_t)0.0;
            scalar_t right = (dR != (scalar_t)0.0) ?
                              cur[r+1] * (t_rd1 - xval) / dR : (scalar_t)0.0;
            nxt[r] = left + right;
        }
        // swap
        scalar_t* tmp = cur; cur = nxt; nxt = tmp;
        // cur now has n_nxt = n_cur-1 entries
    }

    // Write R entries from cur into H
    for (int r = 0; r < R; ++r) {
        H[(b * I + i) * R + r] = cur[r];
    }
}

// ---------------------------------------------------------------------------
// Basis derivative kernel: dH/dx using degree-(K-1) basis
// dB_r^K/dx = K*(B_r^{K-1}/(t_{r+K}-t_r) - B_{r+1}^{K-1}/(t_{r+K+1}-t_{r+1}))
// ---------------------------------------------------------------------------

template <int K, typename scalar_t>
__global__ void basis_derivative_kernel(
    const scalar_t* __restrict__ x,      // [B, I]
    const scalar_t* __restrict__ grid,   // [I, N]
    scalar_t* __restrict__ dH_dx,        // [B, I, R]
    int B, int I, int G
) {
    extern __shared__ char smem_raw[];
    scalar_t* sknots = reinterpret_cast<scalar_t*>(smem_raw);

    const int i = blockIdx.y;
    const int b = blockIdx.x * 32 + threadIdx.x;

    const int N  = G + 2 * K + 1;
    const int R  = G + K;
    const int Km = K - 1;              // degree of lower-order basis
    const int Rm = R + 1;              // G+K = (G+(K-1))+1 basis funcs at Km

    for (int t = threadIdx.x; t < N; t += 32) {
        sknots[t] = grid[i * N + t];
    }
    __syncthreads();

    if (b >= B) return;
    if (K == 0) {
        for (int r = 0; r < R; ++r) dH_dx[(b * I + i) * R + r] = (scalar_t)0.0;
        return;
    }

    scalar_t xval = x[b * I + i];

    // Compute degree-(K-1) basis: Rm entries
    scalar_t buf0[KAN_CUDA_MAX_DEG0_BASIS];
    scalar_t buf1[KAN_CUDA_MAX_DEG0_BASIS];

    int n0 = N - 1; // degree-0 has N-1 entries
    for (int r = 0; r < n0; ++r) {
        scalar_t lo = sknots[r];
        scalar_t hi = sknots[r + 1];
        bool inside = (xval >= lo) && (xval < hi);
        if (r == n0 - 1) inside = inside || (xval == hi);
        buf0[r] = inside ? (scalar_t)1.0 : (scalar_t)0.0;
    }

    scalar_t* cur = buf0;
    scalar_t* nxt = buf1;

    for (int d = 1; d <= Km; ++d) {
        int n_cur = n0 - (d - 1);
        int n_nxt = n_cur - 1;
        for (int r = 0; r < n_nxt; ++r) {
            scalar_t t_r   = sknots[r];
            scalar_t t_rd  = sknots[r + d];
            scalar_t t_r1  = sknots[r + 1];
            scalar_t t_rd1 = sknots[r + d + 1];
            scalar_t dL    = t_rd  - t_r;
            scalar_t dR    = t_rd1 - t_r1;
            scalar_t left  = (dL != (scalar_t)0.0) ?
                              cur[r]   * (xval - t_r)  / dL : (scalar_t)0.0;
            scalar_t right = (dR != (scalar_t)0.0) ?
                              cur[r+1] * (t_rd1 - xval) / dR : (scalar_t)0.0;
            nxt[r] = left + right;
        }
        scalar_t* tmp = cur; cur = nxt; nxt = tmp;
    }
    // cur has Rm entries (degree K-1 basis)

    // Apply derivative formula: R entries
    scalar_t Kf = (scalar_t)K;
    for (int r = 0; r < R; ++r) {
        scalar_t t_r   = sknots[r];
        scalar_t t_rK  = sknots[r + K];
        scalar_t t_r1  = sknots[r + 1];
        scalar_t t_rK1 = sknots[r + K + 1];

        scalar_t dL = t_rK  - t_r;
        scalar_t dR = t_rK1 - t_r1;

        scalar_t term1 = (dL != (scalar_t)0.0) ? cur[r]   / dL : (scalar_t)0.0;
        scalar_t term2 = (dR != (scalar_t)0.0) ? cur[r+1] / dR : (scalar_t)0.0;
        dH_dx[(b * I + i) * R + r] = Kf * (term1 - term2);
    }
}

// ===========================================================================
// Local B-spline evaluation (Phase 7)
//
// For x in knot span [t_m, t_{m+1}) only B_{m-K..m, K} are non-zero
// (k+1 functions).  Each thread:
//   1. locates m by binary search over the (shared-memory) knot row,
//   2. evaluates the K+1 active values with the triangular de Boor scheme
//      (O(K^2) work, registers only),
//   3. writes the dense row H[b,i,:] (zeros + K+1 values) — the GEMM layout
//      is unchanged — and, optionally, the K+1 local derivatives.
//
// Conventions match the dense Cox-de Boor recurrence exactly:
//   * half-open spans [t_r, t_{r+1}), last span closed at t_{N-1},
//   * x outside [t_0, t_{N-1}] gives all-zero basis (and derivative),
//   * zero-length spans contribute zero (repeated knots are allowed).
// ===========================================================================

// Returns the span index m in [0, N-2] with t_m <= x < t_{m+1}
// (x == t_{N-1} -> last non-degenerate span), or -1 if x is outside.
template <typename scalar_t>
__device__ __forceinline__ int find_span(const scalar_t* t, int N, scalar_t x) {
    if (!(x >= t[0]) || x > t[N - 1]) return -1;   // also rejects NaN
    if (x == t[N - 1]) {
        int m = N - 2;
        while (m > 0 && t[m] == t[N - 1]) --m;      // skip zero-length tail spans
        return m;
    }
    int lo = 0, hi = N - 1;                         // invariant t[lo] <= x < t[hi]
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (t[mid] <= x) lo = mid; else hi = mid;
    }
    return lo;
}

// Active basis values Nv[j] = B_{m-K+j, K}(x), j = 0..K, and optionally the
// derivatives Dv[j].  Triangular scheme (Piegl & Tiller, A2.2): every
// contribution of a degree-(d-1) function B_q uses only q's own knots
// t_q..t_{q+d}, so the valid functions (0 <= m-K+j <= R-1) only ever read
// in-range knots.  Window entries that are not real basis functions (outer
// spans) may read out-of-range positions; those reads are clamped and the
// entries are discarded by the caller.
template <int K, typename scalar_t, bool DERIV>
__device__ __forceinline__ void local_basis(const scalar_t* t, int N, int m, scalar_t x,
                                            scalar_t* Nv, scalar_t* Dv) {
    auto T = [&](int idx) { return t[idx < 0 ? 0 : (idx > N - 1 ? N - 1 : idx)]; };
    scalar_t Nd[K + 1];
    Nd[0] = (scalar_t)1;
#pragma unroll
    for (int d = 1; d <= K; ++d) {
        if (DERIV && d == K) {
            // Nd[0..K-1] = B_{m-K+1+j, K-1}.
            // dB_{r,K} = K * (B_{r,K-1}/(t_{r+K}-t_r) - B_{r+1,K-1}/(t_{r+K+1}-t_{r+1}))
#pragma unroll
            for (int j = 0; j <= K; ++j) {
                const int r = m - K + j;
                scalar_t a = (scalar_t)0, b = (scalar_t)0;
                if (j >= 1) {
                    const scalar_t den = T(r + K) - T(r);
                    if (den != (scalar_t)0) a = Nd[j - 1] / den;
                }
                if (j <= K - 1) {
                    const scalar_t den = T(r + K + 1) - T(r + 1);
                    if (den != (scalar_t)0) b = Nd[j] / den;
                }
                Dv[j] = (scalar_t)K * (a - b);
            }
        }
        // Nd[j] = B_{m-d+1+j, d-1}  ->  Nd[j] = B_{m-d+j, d}
        scalar_t saved = (scalar_t)0;
#pragma unroll
        for (int j = 0; j < d; ++j) {
            const scalar_t rgt = T(m + j + 1) - x;
            const scalar_t lft = x - T(m + 1 - d + j);
            const scalar_t dd = rgt + lft;           // t_{m+j+1} - t_{m+1-d+j}
            const scalar_t tmp = (dd != (scalar_t)0) ? Nd[j] / dd : (scalar_t)0;
            Nd[j] = saved + rgt * tmp;
            saved = lft * tmp;
        }
        Nd[d] = saved;
    }
#pragma unroll
    for (int j = 0; j <= K; ++j) Nv[j] = Nd[j];
}

// Which basis kernel the native forward uses (benchmark switch; default local).
static int g_basis_impl = 1;   // 0 = full Cox-de Boor recurrence, 1 = local
void kan_set_basis_impl(int64_t impl) {
    TORCH_CHECK(impl == 0 || impl == 1, "basis impl must be 0 (recurrence) or 1 (local)");
    g_basis_impl = (int)impl;
}
int64_t kan_get_basis_impl() { return g_basis_impl; }

// Dense H[b,i,:] from the local values (zeros outside the active window).
// One thread per (b, i); consecutive threads -> consecutive i -> contiguous
// R-length rows, so stores coalesce.
template <int K, typename scalar_t>
__global__ void basis_local_kernel(const scalar_t* __restrict__ x,
                                   const scalar_t* __restrict__ grid,
                                   scalar_t* __restrict__ H,
                                   int64_t BI, int I, int G) {
    const int N = G + 2 * K + 1, R = G + K;
    for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < BI;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int i = (int)(idx % I);
        const scalar_t* t = grid + (int64_t)i * N;
        const scalar_t xv = x[idx];
        scalar_t* h = H + idx * R;
        const int m = find_span(t, N, xv);
        if (m < 0) {
            for (int r = 0; r < R; ++r) h[r] = (scalar_t)0;
            continue;
        }
        scalar_t Nv[K + 1], Dv[K + 1];
        local_basis<K, scalar_t, false>(t, N, m, xv, Nv, Dv);
        const int r0 = m - K;
        for (int r = 0; r < R; ++r) {
            const int j = r - r0;
            scalar_t v = (scalar_t)0;
#pragma unroll
            for (int jj = 0; jj <= K; ++jj) if (j == jj) v = Nv[jj];
            h[r] = v;
        }
    }
}

// gx[b,i] = sum_r gH[b,i,r] * dB_r/dx  using only the K+1 active derivatives.
template <int K, typename scalar_t>
__global__ void basis_grad_local_kernel(const scalar_t* __restrict__ x,
                                        const scalar_t* __restrict__ grid,
                                        const scalar_t* __restrict__ gH,
                                        scalar_t* __restrict__ gx,
                                        int64_t BI, int I, int G) {
    const int N = G + 2 * K + 1, R = G + K;
    for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < BI;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int i = (int)(idx % I);
        const scalar_t* t = grid + (int64_t)i * N;
        const scalar_t xv = x[idx];
        const int m = find_span(t, N, xv);
        scalar_t acc = (scalar_t)0;
        if (m >= 0) {
            scalar_t Nv[K + 1], Dv[K + 1];
            local_basis<K, scalar_t, true>(t, N, m, xv, Nv, Dv);
            const scalar_t* g = gH + idx * R;
#pragma unroll
            for (int j = 0; j <= K; ++j) {
                const int r = m - K + j;
                if (r >= 0 && r < R) acc += g[r] * Dv[j];
            }
        }
        gx[idx] = acc;
    }
}

static inline int flat_blocks(int64_t n, int threads) {
    int64_t b = (n + threads - 1) / threads;
    return (int)std::min<int64_t>(std::max<int64_t>(b, 1), 65535 * 8);
}

template <typename scalar_t>
static void launch_basis_local(const scalar_t* x, const scalar_t* grid, scalar_t* H,
                               int64_t B, int I, int G, int K, cudaStream_t st) {
    const int64_t BI = B * I;
    const int th = 256, bl = flat_blocks(BI, th);
    switch (K) {
        case 1: basis_local_kernel<1, scalar_t><<<bl, th, 0, st>>>(x, grid, H, BI, I, G); break;
        case 2: basis_local_kernel<2, scalar_t><<<bl, th, 0, st>>>(x, grid, H, BI, I, G); break;
        default: basis_local_kernel<3, scalar_t><<<bl, th, 0, st>>>(x, grid, H, BI, I, G); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
static void launch_basis_grad_local(const scalar_t* x, const scalar_t* grid, const scalar_t* gH,
                                    scalar_t* gx, int64_t B, int I, int G, int K, cudaStream_t st) {
    const int64_t BI = B * I;
    const int th = 256, bl = flat_blocks(BI, th);
    switch (K) {
        case 1: basis_grad_local_kernel<1, scalar_t><<<bl, th, 0, st>>>(x, grid, gH, gx, BI, I, G); break;
        case 2: basis_grad_local_kernel<2, scalar_t><<<bl, th, 0, st>>>(x, grid, gH, gx, BI, I, G); break;
        default: basis_grad_local_kernel<3, scalar_t><<<bl, th, 0, st>>>(x, grid, gH, gx, BI, I, G); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Writes H via the selected implementation.
template <typename scalar_t>
static void basis_forward(const scalar_t* x, const scalar_t* grid, scalar_t* H,
                          int64_t B, int I, int G, int K, cudaStream_t st);
// gx = sum_r gH * dB_r/dx via the selected implementation.
template <typename scalar_t>
static void basis_grad_x(const scalar_t* x, const scalar_t* grid, const scalar_t* gH,
                         scalar_t* gx, int64_t B, int I, int G, int K, cudaStream_t st);

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

template <typename scalar_t>
static void launch_basis_eval(
    const scalar_t* x, const scalar_t* grid, scalar_t* H,
    int B, int I, int G, int K, cudaStream_t stream
) {
    const int N  = G + 2 * K + 1;
    dim3 grid_dim((B + 31) / 32, I);
    dim3 block_dim(32, 1);
    size_t smem = N * sizeof(scalar_t);

    switch (K) {
        case 1:
            basis_eval_kernel<1, scalar_t><<<grid_dim, block_dim, smem, stream>>>(
                x, grid, H, B, I, G);
            break;
        case 2:
            basis_eval_kernel<2, scalar_t><<<grid_dim, block_dim, smem, stream>>>(
                x, grid, H, B, I, G);
            break;
        case 3:
            basis_eval_kernel<3, scalar_t><<<grid_dim, block_dim, smem, stream>>>(
                x, grid, H, B, I, G);
            break;
        default:
            // Fallback: use K=3 template but it won't unroll correctly for K>3.
            // For correctness with higher orders, use the CPU path.
            TORCH_CHECK(false,
                "CUDA kernel only supports spline_order 1,2,3; got ", K,
                ". Use CPU for higher orders.");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
static void launch_basis_derivative(
    const scalar_t* x, const scalar_t* grid, scalar_t* dH_dx,
    int B, int I, int G, int K, cudaStream_t stream
) {
    const int N  = G + 2 * K + 1;
    dim3 grid_dim((B + 31) / 32, I);
    dim3 block_dim(32, 1);
    size_t smem = N * sizeof(scalar_t);

    switch (K) {
        case 1:
            basis_derivative_kernel<1, scalar_t><<<grid_dim, block_dim, smem, stream>>>(
                x, grid, dH_dx, B, I, G);
            break;
        case 2:
            basis_derivative_kernel<2, scalar_t><<<grid_dim, block_dim, smem, stream>>>(
                x, grid, dH_dx, B, I, G);
            break;
        case 3:
            basis_derivative_kernel<3, scalar_t><<<grid_dim, block_dim, smem, stream>>>(
                x, grid, dH_dx, B, I, G);
            break;
        default:
            TORCH_CHECK(false,
                "CUDA derivative kernel only supports spline_order 1,2,3; got ", K);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template <typename scalar_t>
static void basis_forward(const scalar_t* x, const scalar_t* grid, scalar_t* H,
                          int64_t B, int I, int G, int K, cudaStream_t st) {
    if (B == 0) return;
    if (g_basis_impl == 1) launch_basis_local<scalar_t>(x, grid, H, B, I, G, K, st);
    else launch_basis_eval<scalar_t>(x, grid, H, (int)B, I, G, K, st);
}

template <typename scalar_t>
static void basis_grad_x(const scalar_t* x, const scalar_t* grid, const scalar_t* gH,
                         scalar_t* gx, int64_t B, int I, int G, int K, cudaStream_t st) {
    if (B == 0) return;
    launch_basis_grad_local<scalar_t>(x, grid, gH, gx, B, I, G, K, st);
}

// ---------------------------------------------------------------------------
// CUDA Forward
// ---------------------------------------------------------------------------
std::pair<torch::Tensor, torch::Tensor> kan_forward_cuda(
    torch::Tensor x,
    torch::Tensor grid,
    torch::Tensor base_weight,
    torch::Tensor spline_weight,
    torch::Tensor spline_scaler,
    int64_t spline_order,
    bool has_scaler
) {
    TORCH_CHECK(x.is_cuda(),           "kan_forward_cuda: x must be on CUDA");
    TORCH_CHECK(grid.is_cuda(),        "kan_forward_cuda: grid must be on CUDA");
    kan_check_inputs(x, grid, base_weight, spline_weight, spline_scaler,
                     spline_order, has_scaler);
    const bool has_base = kan_has_base(base_weight);

    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream(x.device().index());

    x             = x.contiguous();
    grid          = grid.contiguous();
    spline_weight = spline_weight.contiguous();
    if (has_base) base_weight = base_weight.contiguous();
    if (has_scaler) spline_scaler = spline_scaler.contiguous();

    const int64_t B = x.size(0);
    const int64_t I = x.size(1);
    const int64_t O = spline_weight.size(0);
    const int64_t N = grid.size(1);          // G + 2*K + 1
    const int64_t K = spline_order;
    const int64_t G = N - 2 * K - 1;
    const int64_t R = G + K;                 // number of basis functions

    auto H = torch::empty({B, I, R}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "basis_eval_cuda", [&]() {
        basis_forward<scalar_t>(
            x.data_ptr<scalar_t>(),
            grid.data_ptr<scalar_t>(),
            H.data_ptr<scalar_t>(),
            B, (int)I, (int)G, (int)K,
            stream.stream()
        );
    });

    // effective_weight [O, I, R]
    torch::Tensor eff_weight;
    if (has_scaler) {
        eff_weight = spline_weight * spline_scaler.unsqueeze(2);
    } else {
        eff_weight = spline_weight;
    }

    auto H_flat = H.reshape({B, I * R});
    auto W_flat = eff_weight.reshape({O, I * R});
    auto Y_spline = torch::mm(H_flat, W_flat.t());

    if (!has_base) {
        return {Y_spline, H};
    }
    auto silu_x = torch::nn::functional::silu(x);
    auto Y_base = torch::mm(silu_x, base_weight.t());

    return {Y_base + Y_spline, H};
}

// ---------------------------------------------------------------------------
// CUDA Backward
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
kan_backward_cuda_ex(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor grid,
    torch::Tensor base_weight,
    torch::Tensor spline_weight,
    torch::Tensor spline_scaler,
    torch::Tensor basis,
    int64_t spline_order,
    bool has_scaler,
    bool need_x, bool need_base, bool need_spline
) {
    TORCH_CHECK(x.is_cuda(), "kan_backward_cuda: x must be on CUDA");
    kan_check_inputs(x, grid, base_weight, spline_weight, spline_scaler,
                     spline_order, has_scaler);
    const bool has_base = kan_has_base(base_weight);
    need_base = need_base && has_base;

    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream(x.device().index());

    grad_output   = grad_output.contiguous();
    x             = x.contiguous();
    grid          = grid.contiguous();
    spline_weight = spline_weight.contiguous();
    if (has_base) base_weight = base_weight.contiguous();
    if (has_scaler) spline_scaler = spline_scaler.contiguous();

    const int64_t B = x.size(0);
    const int64_t I = x.size(1);
    const int64_t O = spline_weight.size(0);
    const int64_t N = grid.size(1);
    const int64_t K = spline_order;
    const int64_t G = N - 2 * K - 1;
    const int64_t R = G + K;

    auto& D = grad_output;

    // dW_base = D.T @ silu(x)
    torch::Tensor grad_base;
    if (need_base) {
        grad_base = torch::mm(D.t(), torch::nn::functional::silu(x));
    }

    // eff_weight [O, I, R]
    torch::Tensor eff_weight = has_scaler ? spline_weight * spline_scaler.unsqueeze(2)
                                          : spline_weight;

    torch::Tensor grad_spline, grad_scaler;
    if (need_spline) {
        // DtH [O, I, R]
        TORCH_CHECK(basis.defined() && basis.numel() == B * I * R,
                    "kan_backward_cuda: saved basis missing or wrong size");
        auto DtH = torch::mm(D.t(), basis.contiguous().reshape({B, I * R})).reshape({O, I, R});
        if (has_scaler) {
            grad_spline = spline_scaler.unsqueeze(2) * DtH;
            grad_scaler = (spline_weight * DtH).sum(2);
        } else {
            grad_spline = DtH;
        }
    }

    torch::Tensor grad_x;
    if (need_x) {
        // Spline part: gx[b,i] = sum_r (D @ W)[b,i,r] * dB_r/dx, contracted in the
        // local kernel (only the k+1 active derivatives; dH/dx never stored).
        auto DeffW = torch::mm(D, eff_weight.reshape({O, I * R}));      // [B, I*R]
        if (g_basis_impl == 1) {
            grad_x = torch::empty({B, I}, x.options());
            AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "basis_grad_x_cuda", [&]() {
                basis_grad_x<scalar_t>(x.data_ptr<scalar_t>(), grid.data_ptr<scalar_t>(),
                                       DeffW.data_ptr<scalar_t>(), grad_x.data_ptr<scalar_t>(),
                                       B, (int)I, (int)G, (int)K, stream.stream());
            });
        } else {
            // Legacy: materialise dH/dx [B, I, R] with the recurrence kernel.
            auto dH_dx = torch::empty({B, I, R}, x.options());
            AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "basis_derivative_cuda", [&]() {
                launch_basis_derivative<scalar_t>(x.data_ptr<scalar_t>(), grid.data_ptr<scalar_t>(),
                                                  dH_dx.data_ptr<scalar_t>(), (int)B, (int)I,
                                                  (int)G, (int)K, stream.stream());
            });
            grad_x = (DeffW.reshape({B, I, R}) * dH_dx).sum(2);
        }
        if (has_base) {
            // silu'(x) = sigmoid(x)*(1 + x*(1-sigmoid(x)))
            auto sig_x = torch::sigmoid(x);
            grad_x.add_(torch::mm(D, base_weight) * (sig_x * (1.0 + x * (1.0 - sig_x))));
        }
    }

    return {grad_x, grad_base, grad_spline, grad_scaler};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
kan_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor grid,
    torch::Tensor base_weight,
    torch::Tensor spline_weight,
    torch::Tensor spline_scaler,
    torch::Tensor basis,
    int64_t spline_order,
    bool has_scaler
) {
    return kan_backward_cuda_ex(grad_output, x, grid, base_weight, spline_weight,
                                spline_scaler, basis, spline_order, has_scaler,
                                true, true, true);
}

static void check_basis_args(const torch::Tensor& x, const torch::Tensor& grid, int64_t k) {
    TORCH_CHECK(x.is_cuda() && grid.is_cuda(), "kan_basis: x and grid must be CUDA tensors");
    TORCH_CHECK(x.dim() == 2 && grid.dim() == 2 && grid.size(0) == x.size(1),
                "kan_basis: x must be [B, I] and grid [I, G+2k+1]");
    TORCH_CHECK(k >= 1 && k <= 3, "kan_basis: spline_order must be 1, 2 or 3");
    TORCH_CHECK(x.scalar_type() == grid.scalar_type(), "kan_basis: dtype mismatch");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat64,
                "kan_basis: only FP32/FP64 supported");
    const int64_t G = grid.size(1) - 2 * k - 1;
    TORCH_CHECK(G >= 1, "kan_basis: grid too short");
    TORCH_CHECK(G + 2 * k <= KAN_CUDA_MAX_DEG0_BASIS,
        "CUDA kernel supports G + 2k <= ", KAN_CUDA_MAX_DEG0_BASIS, " (got G=", G, ", k=", k, ")");
    TORCH_CHECK(x.size(1) <= KAN_CUDA_MAX_IN_FEATURES, "kan_basis: too many input features");
    TORCH_CHECK(x.size(0) * x.size(1) * (G + 2 * k) <= std::numeric_limits<int>::max(),
                "kan_basis: B*I*(G+2k) too large for 32-bit indexing");
}

torch::Tensor kan_basis_cuda(torch::Tensor x, torch::Tensor grid, int64_t k) {
    check_basis_args(x, grid, k);
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
    x = x.contiguous(); grid = grid.contiguous();
    const int64_t B = x.size(0), I = x.size(1), G = grid.size(1) - 2 * k - 1;
    auto H = torch::empty({B, I, G + k}, x.options());
    if (B == 0) return H;
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "kan_basis_cuda", [&]() {
        basis_forward<scalar_t>(x.data_ptr<scalar_t>(), grid.data_ptr<scalar_t>(),
                                H.data_ptr<scalar_t>(), B, (int)I, (int)G, (int)k,
                                stream.stream());
    });
    return H;
}

torch::Tensor kan_basis_grad_cuda(torch::Tensor x, torch::Tensor grid, int64_t k, torch::Tensor gH) {
    check_basis_args(x, grid, k);
    const int64_t B = x.size(0), I = x.size(1), G = grid.size(1) - 2 * k - 1;
    TORCH_CHECK(gH.dim() == 3 && gH.size(0) == B && gH.size(1) == I && gH.size(2) == G + k,
                "kan_basis_grad: gH must be [B, I, G+k]");
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream(x.device().index());
    x = x.contiguous(); grid = grid.contiguous(); gH = gH.contiguous().to(x.scalar_type());
    auto gx = torch::empty({B, I}, x.options());
    if (B == 0) return gx;
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "kan_basis_grad_cuda", [&]() {
        basis_grad_x<scalar_t>(x.data_ptr<scalar_t>(), grid.data_ptr<scalar_t>(),
                               gH.data_ptr<scalar_t>(), gx.data_ptr<scalar_t>(),
                               B, (int)I, (int)G, (int)k, stream.stream());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return gx;
}

