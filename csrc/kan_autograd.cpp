#include "kan_ops.h"
#include <torch/extension.h>
#include <torch/autograd.h>

// ---------------------------------------------------------------------------
// KANFunction: custom autograd Function
//
// PyTorch C++ Function::apply only accepts Tensor arguments.  Non-tensor
// configuration (spline_order, has_scaler) is baked into the closure via
// kan_apply rather than being passed through apply().
// ---------------------------------------------------------------------------

struct KANFunction : public torch::autograd::Function<KANFunction> {
    // Exposed via a wrapper; see kan_apply below.
    // forward() receives (x, grid, base_weight, spline_weight, spline_scaler)
    // plus two extra 0-d scalar tensors encoding spline_order and has_scaler.

    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor x,
        torch::Tensor grid,
        torch::Tensor base_weight,
        torch::Tensor spline_weight,
        torch::Tensor spline_scaler,   // may be empty when !has_scaler
        torch::Tensor order_t,         // scalar int64 tensor carrying spline_order
        torch::Tensor scaler_t         // scalar int64 tensor carrying has_scaler (0/1)
    ) {
        int64_t spline_order = order_t.item<int64_t>();
        bool    has_scaler   = scaler_t.item<int64_t>() != 0;

        torch::Tensor output, basis;
        if (x.is_cuda()) {
            auto r = kan_forward_cuda(x, grid, base_weight, spline_weight,
                                      spline_scaler, spline_order, has_scaler);
            output = r.first; basis = r.second;
        } else {
            auto r = kan_forward_cpu(x, grid, base_weight, spline_weight,
                                     spline_scaler, spline_order, has_scaler);
            output = r.first; basis = r.second;
        }

        ctx->save_for_backward({x, grid, base_weight, spline_weight,
                                 has_scaler ? spline_scaler : torch::Tensor(),
                                 basis, order_t, scaler_t});
        return output;
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        auto s             = ctx->get_saved_variables();
        auto x              = s[0];
        auto grid           = s[1];
        auto base_weight    = s[2];
        auto spline_weight  = s[3];
        auto spline_scaler  = s[4];
        auto basis          = s[5];
        int64_t spline_order = s[6].item<int64_t>();
        bool    has_scaler   = s[7].item<int64_t>() != 0;

        auto D = grad_outputs[0];

        torch::Tensor gx, gbase, gspline, gscaler;
        if (torch::GradMode::is_enabled()) {
            // create_graph=True: the backward itself is being recorded.  The
            // native kernels (saved basis buffer, dH/dx kernel) are not
            // differentiable, so use the composite ATen formulation whose
            // graph carries the exact spline curvature.  This is what makes
            // u_xx / PDE losses correct on the CUDA backend.
            auto g = kan_backward_composite(D, x, grid, base_weight, spline_weight,
                                            spline_scaler, spline_order, has_scaler);
            gx = std::get<0>(g); gbase = std::get<1>(g);
            gspline = std::get<2>(g); gscaler = std::get<3>(g);
        } else if (x.is_cuda()) {
#ifdef KAN_WITH_CUDA
            // Only compute what autograd asks for (e.g. no grad_x for a
            // first layer whose input does not require grad).
            const bool need_x      = ctx->needs_input_grad(0);
            const bool need_base   = ctx->needs_input_grad(2);
            const bool need_spline = ctx->needs_input_grad(3) ||
                                     (has_scaler && ctx->needs_input_grad(4));
            auto g = kan_backward_cuda_ex(D, x, grid, base_weight, spline_weight,
                                          spline_scaler, basis, spline_order, has_scaler,
                                          need_x, need_base, need_spline);
#else
            auto g = kan_backward_cuda(D, x, grid, base_weight, spline_weight,
                                       spline_scaler, basis, spline_order, has_scaler);
#endif
            gx = std::get<0>(g); gbase = std::get<1>(g);
            gspline = std::get<2>(g); gscaler = std::get<3>(g);
        } else {
            auto g = kan_backward_cpu(D, x, grid, base_weight, spline_weight,
                                      spline_scaler, basis, spline_order, has_scaler);
            gx = std::get<0>(g); gbase = std::get<1>(g);
            gspline = std::get<2>(g); gscaler = std::get<3>(g);
        }

        // Return in same order as forward inputs:
        // x, grid, base_weight, spline_weight, spline_scaler, order_t, scaler_t
        return {
            gx,
            torch::Tensor(),                              // grid – not trainable
            kan_has_base(base_weight) ? gbase : torch::Tensor(),
            gspline,
            has_scaler ? gscaler : torch::Tensor(),
            torch::Tensor(),                              // order_t – not trainable
            torch::Tensor(),                              // scaler_t – not trainable
        };
    }
};

// ---------------------------------------------------------------------------
// Python-callable entry point
// ---------------------------------------------------------------------------

torch::Tensor kan_apply(
    torch::Tensor x,
    torch::Tensor grid,
    torch::Tensor base_weight,
    torch::Tensor spline_weight,
    torch::Tensor spline_scaler,
    int64_t spline_order,
    bool has_scaler
) {
    // Encode non-tensor config as scalar CPU int64 tensors so Function::apply
    // can carry them without special casing.
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto order_t  = torch::tensor(spline_order,          opts);
    auto scaler_t = torch::tensor((int64_t)has_scaler,   opts);

    return KANFunction::apply(
        x, grid, base_weight, spline_weight,
        spline_scaler, order_t, scaler_t);
}
