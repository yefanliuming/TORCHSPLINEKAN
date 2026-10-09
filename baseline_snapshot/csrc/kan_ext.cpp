#include "kan_ops.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// ---------------------------------------------------------------------------
// Device-dispatching wrappers used by both TORCH_LIBRARY and pybind11 paths
// ---------------------------------------------------------------------------

static std::tuple<torch::Tensor, torch::Tensor> kan_forward_dispatch(
    torch::Tensor x,
    torch::Tensor grid,
    torch::Tensor base_weight,
    torch::Tensor spline_weight,
    torch::Tensor spline_scaler,
    int64_t spline_order,
    bool has_scaler
) {
    TORCH_CHECK(x.dim() == 2, "x must be 2-D [B, I], got ", x.dim(), "D");
    TORCH_CHECK(grid.dim() == 2, "grid must be 2-D [I, G+2k+1]");
    TORCH_CHECK(base_weight.dim() == 2, "base_weight must be 2-D [O, I]");
    TORCH_CHECK(spline_weight.dim() == 3, "spline_weight must be 3-D [O, I, R]");
    TORCH_CHECK(spline_order >= 1, "spline_order must be >= 1");

    if (x.is_cuda()) {
        auto result = kan_forward_cuda(
            x, grid, base_weight, spline_weight,
            spline_scaler, spline_order, has_scaler);
        return {result.first, result.second};
    } else {
        auto result = kan_forward_cpu(
            x, grid, base_weight, spline_weight,
            spline_scaler, spline_order, has_scaler);
        return {result.first, result.second};
    }
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
kan_backward_dispatch(
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
    if (x.is_cuda()) {
        return kan_backward_cuda(
            grad_output, x, grid, base_weight, spline_weight,
            spline_scaler, basis, spline_order, has_scaler);
    } else {
        return kan_backward_cpu(
            grad_output, x, grid, base_weight, spline_weight,
            spline_scaler, basis, spline_order, has_scaler);
    }
}

// ---------------------------------------------------------------------------
// TORCH_LIBRARY op registration
// ---------------------------------------------------------------------------

TORCH_LIBRARY(spline_kan, m) {
    m.def(
        "kan_forward(Tensor x, Tensor grid, Tensor base_weight, "
        "Tensor spline_weight, Tensor spline_scaler, int spline_order, "
        "bool has_scaler) -> (Tensor output, Tensor basis)"
    );
    m.def(
        "kan_backward(Tensor grad_output, Tensor x, Tensor grid, "
        "Tensor base_weight, Tensor spline_weight, Tensor spline_scaler, "
        "Tensor basis, int spline_order, bool has_scaler) "
        "-> (Tensor grad_x, Tensor grad_base_weight, "
        "Tensor grad_spline_weight, Tensor grad_spline_scaler)"
    );
}

// CPU implementation
TORCH_LIBRARY_IMPL(spline_kan, CPU, m) {
    m.impl("kan_forward",
        [](torch::Tensor x, torch::Tensor grid,
           torch::Tensor base_weight, torch::Tensor spline_weight,
           torch::Tensor spline_scaler, int64_t spline_order, bool has_scaler)
        -> std::tuple<torch::Tensor, torch::Tensor> {
            auto r = kan_forward_cpu(x, grid, base_weight, spline_weight,
                                     spline_scaler, spline_order, has_scaler);
            return {r.first, r.second};
        });
    m.impl("kan_backward",
        [](torch::Tensor grad_output, torch::Tensor x, torch::Tensor grid,
           torch::Tensor base_weight, torch::Tensor spline_weight,
           torch::Tensor spline_scaler, torch::Tensor basis,
           int64_t spline_order, bool has_scaler)
        -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> {
            return kan_backward_cpu(grad_output, x, grid, base_weight,
                                    spline_weight, spline_scaler, basis,
                                    spline_order, has_scaler);
        });
}

// CUDA implementation
TORCH_LIBRARY_IMPL(spline_kan, CUDA, m) {
    m.impl("kan_forward",
        [](torch::Tensor x, torch::Tensor grid,
           torch::Tensor base_weight, torch::Tensor spline_weight,
           torch::Tensor spline_scaler, int64_t spline_order, bool has_scaler)
        -> std::tuple<torch::Tensor, torch::Tensor> {
            auto r = kan_forward_cuda(x, grid, base_weight, spline_weight,
                                      spline_scaler, spline_order, has_scaler);
            return {r.first, r.second};
        });
    m.impl("kan_backward",
        [](torch::Tensor grad_output, torch::Tensor x, torch::Tensor grid,
           torch::Tensor base_weight, torch::Tensor spline_weight,
           torch::Tensor spline_scaler, torch::Tensor basis,
           int64_t spline_order, bool has_scaler)
        -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> {
            return kan_backward_cuda(grad_output, x, grid, base_weight,
                                     spline_weight, spline_scaler, basis,
                                     spline_order, has_scaler);
        });
}

// Meta / fake implementation for torch.compile and tracing
TORCH_LIBRARY_IMPL(spline_kan, Meta, m) {
    m.impl("kan_forward",
        [](torch::Tensor x, torch::Tensor grid,
           torch::Tensor base_weight, torch::Tensor spline_weight,
           torch::Tensor spline_scaler, int64_t spline_order, bool has_scaler)
        -> std::tuple<torch::Tensor, torch::Tensor> {
            const int64_t B = x.size(0);
            const int64_t O = base_weight.size(0);
            const int64_t I = x.size(1);
            const int64_t N = grid.size(1);
            const int64_t R = N - spline_order - 1;
            auto output = x.new_empty({B, O});
            auto basis  = x.new_empty({B, I, R});
            return {output, basis};
        });
    m.impl("kan_backward",
        [](torch::Tensor grad_output, torch::Tensor x, torch::Tensor grid,
           torch::Tensor base_weight, torch::Tensor spline_weight,
           torch::Tensor spline_scaler, torch::Tensor basis,
           int64_t spline_order, bool has_scaler)
        -> std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> {
            auto grad_x             = x.new_empty(x.sizes());
            auto grad_base_weight   = base_weight.new_empty(base_weight.sizes());
            auto grad_spline_weight = spline_weight.new_empty(spline_weight.sizes());
            auto grad_spline_scaler = has_scaler
                ? spline_scaler.new_empty(spline_scaler.sizes())
                : torch::Tensor();
            return {grad_x, grad_base_weight, grad_spline_weight, grad_spline_scaler};
        });
}

// ---------------------------------------------------------------------------
// pybind11 bindings for backward compatibility
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "torch-spline-kan C++/CUDA extension";

    m.def("kan_forward",
        &kan_forward_dispatch,
        "B-spline KAN forward (CPU/CUDA dispatch)",
        py::arg("x"),
        py::arg("grid"),
        py::arg("base_weight"),
        py::arg("spline_weight"),
        py::arg("spline_scaler"),
        py::arg("spline_order"),
        py::arg("has_scaler"));

    m.def("kan_backward",
        &kan_backward_dispatch,
        "B-spline KAN backward (CPU/CUDA dispatch)",
        py::arg("grad_output"),
        py::arg("x"),
        py::arg("grid"),
        py::arg("base_weight"),
        py::arg("spline_weight"),
        py::arg("spline_scaler"),
        py::arg("basis"),
        py::arg("spline_order"),
        py::arg("has_scaler"));

    m.def("kan_apply",
        &kan_apply,
        "B-spline KAN forward with autograd (KANFunction)",
        py::arg("x"),
        py::arg("grid"),
        py::arg("base_weight"),
        py::arg("spline_weight"),
        py::arg("spline_scaler"),
        py::arg("spline_order"),
        py::arg("has_scaler"));
}
