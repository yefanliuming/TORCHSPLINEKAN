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
    scalar_t buf0[71]; // degree-d basis (max N-1 = 70 entries, 0-indexed)
    scalar_t buf1[71];

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
    scalar_t buf0[72];
    scalar_t buf1[72];

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
    TORCH_CHECK(base_weight.is_cuda(), "kan_forward_cuda: base_weight must be on CUDA");

    TORCH_CHECK(x.scalar_type() == torch::kFloat32 ||
                x.scalar_type() == torch::kFloat64,
        "kan_forward_cuda: only FP32 and FP64 are supported; "
        "use FP32 or FP64 for spline computation");

    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream(x.device().index());

    x             = x.contiguous();
    grid          = grid.contiguous();
    base_weight   = base_weight.contiguous();
    spline_weight = spline_weight.contiguous();
    if (has_scaler) spline_scaler = spline_scaler.contiguous();

    const int64_t B = x.size(0);
    const int64_t I = x.size(1);
    const int64_t O = base_weight.size(0);
    const int64_t N = grid.size(1);          // G + 2*K + 1
    const int64_t K = spline_order;
    const int64_t G = N - 2 * K - 1;
    const int64_t R = G + K;                 // number of basis functions

    TORCH_CHECK(R == spline_weight.size(2),
        "R mismatch: computed R=", R, " but spline_weight.size(2)=", spline_weight.size(2));
    TORCH_CHECK(K >= 1 && K <= 3,
        "CUDA path supports spline_order 1,2,3; got ", K);

    auto H = torch::empty({B, I, R}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "basis_eval_cuda", [&]() {
        launch_basis_eval<scalar_t>(
            x.data_ptr<scalar_t>(),
            grid.data_ptr<scalar_t>(),
            H.data_ptr<scalar_t>(),
            (int)B, (int)I, (int)G, (int)K,
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

    auto silu_x = torch::nn::functional::silu(x);
    auto Y_base = torch::mm(silu_x, base_weight.t());

    return {Y_base + Y_spline, H};
}

// ---------------------------------------------------------------------------
// CUDA Backward
// ---------------------------------------------------------------------------
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
    TORCH_CHECK(x.is_cuda(), "kan_backward_cuda: x must be on CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 ||
                x.scalar_type() == torch::kFloat64,
        "kan_backward_cuda: only FP32 and FP64 are supported; "
        "use FP32 or FP64 for spline computation");

    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream(x.device().index());

    grad_output   = grad_output.contiguous();
    x             = x.contiguous();
    grid          = grid.contiguous();
    base_weight   = base_weight.contiguous();
    spline_weight = spline_weight.contiguous();
    basis         = basis.contiguous();
    if (has_scaler) spline_scaler = spline_scaler.contiguous();

    const int64_t B = x.size(0);
    const int64_t I = x.size(1);
    const int64_t O = base_weight.size(0);
    const int64_t N = grid.size(1);
    const int64_t K = spline_order;
    const int64_t G = N - 2 * K - 1;
    const int64_t R = G + K;

    auto& D = grad_output;
    auto& H = basis;

    // dW_base = D.T @ silu(x)
    auto silu_x    = torch::nn::functional::silu(x);
    auto grad_base = torch::mm(D.t(), silu_x);

    // eff_weight [O, I, R]
    torch::Tensor eff_weight;
    if (has_scaler) {
        eff_weight = spline_weight * spline_scaler.unsqueeze(2);
    } else {
        eff_weight = spline_weight;
    }

    // DtH [O, I, R]
    auto H_flat = H.reshape({B, I * R});
    auto DtH    = torch::mm(D.t(), H_flat).reshape({O, I, R});

    torch::Tensor grad_spline, grad_scaler;
    if (has_scaler) {
        grad_spline = spline_scaler.unsqueeze(2) * DtH;
        grad_scaler = (spline_weight * DtH).sum(2);
    } else {
        grad_spline = DtH;
        grad_scaler = torch::Tensor();
    }

    // dX from base: silu'(x) = sigmoid(x)*(1 + x*(1-sigmoid(x)))
    auto sig_x     = torch::sigmoid(x);
    auto silu_grad = sig_x * (1.0 + x * (1.0 - sig_x));
    auto Dbase     = torch::mm(D, base_weight);
    auto dX_base   = Dbase * silu_grad;

    // dH/dx via CUDA kernel
    auto dH_dx = torch::empty({B, I, R}, x.options());
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "basis_derivative_cuda", [&]() {
        launch_basis_derivative<scalar_t>(
            x.data_ptr<scalar_t>(),
            grid.data_ptr<scalar_t>(),
            dH_dx.data_ptr<scalar_t>(),
            (int)B, (int)I, (int)G, (int)K,
            stream.stream()
        );
    });

    // dX from spline: (D @ eff_weight.reshape(O, I*R)).reshape(B,I,R) * dH_dx -> sum over R
    auto DeffW    = torch::mm(D, eff_weight.reshape({O, I * R})).reshape({B, I, R});
    auto dX_spline = (DeffW * dH_dx).sum(2);

    auto grad_x = dX_base + dX_spline;

    return {grad_x, grad_base, grad_spline, grad_scaler};
}
