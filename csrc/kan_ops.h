#pragma once

#include <torch/extension.h>
#include <tuple>
#include <utility>
#include <limits>

// ---------------------------------------------------------------------------
// Shared limits / validation
// ---------------------------------------------------------------------------

// The CUDA basis kernels keep the degree-0 stage (G + 2k values) in a fixed
// per-thread local array.  Configurations beyond this bound are rejected
// before any kernel launch.
constexpr int64_t KAN_CUDA_MAX_DEG0_BASIS = 71;
// gridDim.y limit used for the input-feature axis of the basis kernels.
constexpr int64_t KAN_CUDA_MAX_IN_FEATURES = 65535;

// A layer has a base branch iff base_weight is a non-empty tensor.  An empty
// base_weight means the caller evaluates the base branch itself (used for
// base activations other than SiLU).
inline bool kan_has_base(const torch::Tensor& base_weight) {
    return base_weight.defined() && base_weight.numel() > 0;
}

// Validates shapes / dtypes / devices of the forward inputs.
inline void kan_check_inputs(
    const torch::Tensor& x,
    const torch::Tensor& grid,
    const torch::Tensor& base_weight,
    const torch::Tensor& spline_weight,
    const torch::Tensor& spline_scaler,
    int64_t spline_order,
    bool has_scaler
) {
    TORCH_CHECK(x.dim() == 2, "x must be 2-D [B, I], got ", x.dim(), "-D");
    TORCH_CHECK(grid.dim() == 2, "grid must be 2-D [I, G+2k+1]");
    TORCH_CHECK(spline_weight.dim() == 3, "spline_weight must be 3-D [O, I, R]");
    TORCH_CHECK(spline_order >= 1 && spline_order <= 3,
        "spline_order must be 1, 2 or 3; got ", spline_order);
    const int64_t I = x.size(1);
    const int64_t O = spline_weight.size(0);
    const int64_t N = grid.size(1);
    const int64_t G = N - 2 * spline_order - 1;
    TORCH_CHECK(G >= 1, "grid has ", N, " knots, too few for spline_order ", spline_order);
    TORCH_CHECK(grid.size(0) == I, "grid.size(0)=", grid.size(0), " must equal in_features=", I);
    TORCH_CHECK(spline_weight.size(1) == I, "spline_weight.size(1) must equal in_features=", I);
    TORCH_CHECK(spline_weight.size(2) == G + spline_order,
        "spline_weight.size(2)=", spline_weight.size(2), " but grid implies R=G+k=", G + spline_order);
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kFloat64,
        "only FP32 and FP64 inputs are supported, got ", x.scalar_type());
    TORCH_CHECK(grid.scalar_type() == x.scalar_type() &&
                spline_weight.scalar_type() == x.scalar_type(),
        "x, grid and spline_weight must share a dtype");
    TORCH_CHECK(grid.device() == x.device() && spline_weight.device() == x.device(),
        "x, grid and spline_weight must be on the same device");
    if (kan_has_base(base_weight)) {
        TORCH_CHECK(base_weight.dim() == 2 && base_weight.size(0) == O && base_weight.size(1) == I,
            "base_weight must have shape [O, I] = [", O, ", ", I, "]");
        TORCH_CHECK(base_weight.scalar_type() == x.scalar_type() && base_weight.device() == x.device(),
            "base_weight must match x in dtype and device");
    }
    if (has_scaler) {
        TORCH_CHECK(spline_scaler.dim() == 2 && spline_scaler.size(0) == O && spline_scaler.size(1) == I,
            "spline_scaler must have shape [O, I] = [", O, ", ", I, "]");
        TORCH_CHECK(spline_scaler.scalar_type() == x.scalar_type() && spline_scaler.device() == x.device(),
            "spline_scaler must match x in dtype and device");
    }
    if (x.is_cuda()) {
        TORCH_CHECK(G + 2 * spline_order <= KAN_CUDA_MAX_DEG0_BASIS,
            "CUDA kernel supports G + 2k <= ", KAN_CUDA_MAX_DEG0_BASIS,
            " (got G=", G, ", k=", spline_order, "); use backend='reference' for larger grids");
        TORCH_CHECK(I <= KAN_CUDA_MAX_IN_FEATURES,
            "CUDA kernel supports in_features <= ", KAN_CUDA_MAX_IN_FEATURES, ", got ", I);
        TORCH_CHECK(x.size(0) * I * (N - 1) <= std::numeric_limits<int>::max(),
            "CUDA kernel uses 32-bit indexing; B*I*(G+2k) too large");
    }
}

// ---------------------------------------------------------------------------
// CPU implementations (kan_cpu.cpp)
// ---------------------------------------------------------------------------

std::pair<torch::Tensor, torch::Tensor> kan_forward_cpu(
    torch::Tensor x,
    torch::Tensor grid,
    torch::Tensor base_weight,
    torch::Tensor spline_weight,
    torch::Tensor spline_scaler,
    int64_t spline_order,
    bool has_scaler
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
kan_backward_cpu(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor grid,
    torch::Tensor base_weight,
    torch::Tensor spline_weight,
    torch::Tensor spline_scaler,
    torch::Tensor basis,
    int64_t spline_order,
    bool has_scaler
);

// ---------------------------------------------------------------------------
// CUDA implementations (kan_cuda.cu)
// ---------------------------------------------------------------------------

std::pair<torch::Tensor, torch::Tensor> kan_forward_cuda(
    torch::Tensor x,
    torch::Tensor grid,
    torch::Tensor base_weight,
    torch::Tensor spline_weight,
    torch::Tensor spline_scaler,
    int64_t spline_order,
    bool has_scaler
);

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
);

// Device-agnostic backward built only from differentiable ATen ops (the basis
// is recomputed from x).  Used when the backward itself is being recorded
// (create_graph=True) so that higher-order derivatives are exact.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
kan_backward_composite(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor grid,
    torch::Tensor base_weight,
    torch::Tensor spline_weight,
    torch::Tensor spline_scaler,
    int64_t spline_order,
    bool has_scaler
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
kan_backward_cuda_ex(
    torch::Tensor grad_output, torch::Tensor x, torch::Tensor grid,
    torch::Tensor base_weight, torch::Tensor spline_weight, torch::Tensor spline_scaler,
    torch::Tensor basis, int64_t spline_order, bool has_scaler,
    bool need_x, bool need_base, bool need_spline);
void kan_set_basis_impl(int64_t impl);
int64_t kan_get_basis_impl();

// Native building blocks (CUDA only) used by the fused regularisation path.
torch::Tensor kan_basis_cuda(torch::Tensor x, torch::Tensor grid, int64_t k);
torch::Tensor kan_basis_grad_cuda(torch::Tensor x, torch::Tensor grid, int64_t k, torch::Tensor gH);
torch::Tensor edge_abs_mean_cuda(torch::Tensor s, torch::Tensor H,
                                 torch::Tensor bw, torch::Tensor W);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
edge_abs_mean_backward_cuda(torch::Tensor grad_a, torch::Tensor s, torch::Tensor H,
                            torch::Tensor bw, torch::Tensor W,
                            bool need_s, bool need_H, bool need_bw, bool need_W);

// ---------------------------------------------------------------------------
// Python-callable entry point wrapping KANFunction (kan_autograd.cpp)
// ---------------------------------------------------------------------------

torch::Tensor kan_apply(
    torch::Tensor x,
    torch::Tensor grid,
    torch::Tensor base_weight,
    torch::Tensor spline_weight,
    torch::Tensor spline_scaler,
    int64_t spline_order,
    bool has_scaler
);
