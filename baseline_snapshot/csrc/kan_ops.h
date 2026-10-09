#pragma once

#include <torch/extension.h>
#include <tuple>
#include <utility>

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
