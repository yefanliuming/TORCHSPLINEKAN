#include "kan_ops.h"
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <torch/torch.h>

// ---------------------------------------------------------------------------
// Helper: Cox-de Boor B-spline basis evaluation (CPU, ATen ops)
// grid: [I, G+2k+1], x: [B, I]
// Returns H: [B, I, R] where R = G+k (number of basis functions of degree k)
// ---------------------------------------------------------------------------
static torch::Tensor bspline_basis(
    const torch::Tensor& x,    // [B, I]
    const torch::Tensor& grid, // [I, N] where N = G+2k+1
    int64_t k
) {
    const int64_t B = x.size(0);
    const int64_t I = x.size(1);
    const int64_t N = grid.size(1);   // G+2k+1
    const int64_t R = N - k - 1;      // G+k  (number of basis funcs)

    // x_exp: [B, I, 1]
    auto x_exp = x.unsqueeze(2);

    // grid expanded: [1, I, N]  ->  broadcast with [B, I, N]
    auto g = grid.unsqueeze(0); // [1, I, N]

    // Degree-0 basis: B_r^0 = 1 if grid[i,r] <= x[b,i] < grid[i,r+1]
    // Shape: [B, I, N-1]
    auto g_left  = g.slice(2, 0, N - 1); // [1, I, N-1]
    auto g_right = g.slice(2, 1, N);     // [1, I, N-1]

    // [B, I, N-1]
    auto b0 = ((x_exp >= g_left) & (x_exp < g_right)).to(x.dtype());

    // Special case: include the right endpoint in the last interval for the
    // rightmost basis function at degree 0.
    // b0[b, i, N-2] |= (x[b,i] == grid[i, N-1])
    auto last_mask = (x_exp == g.slice(2, N - 1, N)); // [B, I, 1]
    b0.slice(2, N - 2, N - 1) = b0.slice(2, N - 2, N - 1) + last_mask.to(x.dtype());
    b0 = b0.clamp(0.0, 1.0);

    // Degree-d recurrence
    // After degree d, the number of basis functions is N-1-d
    auto bd = b0; // [B, I, N-1]

    for (int64_t d = 1; d <= k; ++d) {
        // bd has shape [B, I, N-d]
        int64_t n_basis = N - d; // current number of basis functions

        // Left term: B_{r}^{d-1} * (x - t_r) / (t_{r+d} - t_r)
        // t_r  = grid[:, r],    shape [1, I, n_basis]
        // t_rd = grid[:, r+d],  shape [1, I, n_basis]
        auto t_r  = g.slice(2, 0,   n_basis);       // [1, I, n_basis]
        auto t_rd = g.slice(2, d,   n_basis + d);   // [1, I, n_basis]
        auto denom_l = (t_rd - t_r);                // [1, I, n_basis]
        // zero-safe division: where denom==0, set to 0
        auto safe_denom_l = torch::where(denom_l == 0, torch::ones_like(denom_l), denom_l);
        auto left = bd * (x_exp - t_r) / safe_denom_l;
        left = torch::where(denom_l == 0, torch::zeros_like(left), left);

        // Right term: B_{r+1}^{d-1} * (t_{r+d+1} - x) / (t_{r+d+1} - t_{r+1})
        // bd slice [r+1 .. n_basis] -> [B, I, n_basis-1]
        auto bd_r1   = bd.slice(2, 1, n_basis);         // [B, I, n_basis-1]
        auto t_r1    = g.slice(2, 1,   n_basis);         // [1, I, n_basis-1]
        auto t_rd1   = g.slice(2, d+1, n_basis + d);     // [1, I, n_basis-1]
        auto denom_r = (t_rd1 - t_r1);
        auto safe_denom_r = torch::where(denom_r == 0, torch::ones_like(denom_r), denom_r);
        auto right = bd_r1 * (t_rd1 - x_exp) / safe_denom_r;
        right = torch::where(denom_r == 0, torch::zeros_like(right), right);

        // New bd has n_basis-1 entries
        // left: [B, I, n_basis], right: [B, I, n_basis-1]
        // combine: new_bd[r] = left[r] + right[r]  for r=0..n_basis-2
        auto new_bd = left.slice(2, 0, n_basis - 1) + right;
        bd = new_bd; // [B, I, n_basis-1] = [B, I, N-1-d]
    }

    // bd is now [B, I, R] where R = N-1-k = G+k
    TORCH_CHECK(bd.size(2) == R,
        "bspline_basis: expected R=", R, " but got ", bd.size(2));
    return bd.contiguous();
}

// ---------------------------------------------------------------------------
// Helper: B-spline basis derivative dH/dx via recurrence
// Returns dH_dx: [B, I, R]
// ---------------------------------------------------------------------------
static torch::Tensor bspline_basis_derivative(
    const torch::Tensor& x,    // [B, I]
    const torch::Tensor& grid, // [I, N]
    int64_t k
) {
    if (k == 0) {
        // Derivative of degree-0 basis is 0 (piecewise constant)
        const int64_t B = x.size(0);
        const int64_t I = x.size(1);
        const int64_t N = grid.size(1);
        const int64_t R = N - k - 1;
        return torch::zeros({B, I, R}, x.options());
    }

    // dB_r^k/dx = k * (B_r^{k-1} / (t_{r+k} - t_r) - B_{r+1}^{k-1} / (t_{r+k+1} - t_{r+1}))
    // We need the degree-(k-1) basis to compute this
    auto H_km1 = bspline_basis(x, grid, k - 1); // [B, I, R+1]

    const int64_t N = grid.size(1);
    const int64_t R = N - k - 1;
    auto g = grid.unsqueeze(0); // [1, I, N]

    // For r=0..R-1:
    // term1[r] = H_km1[r] / (t_{r+k} - t_r)      -> [B, I, R]
    auto t_r   = g.slice(2, 0, R);       // [1, I, R]
    auto t_rk  = g.slice(2, k, R + k);   // [1, I, R]
    auto denom1 = (t_rk - t_r);
    auto safe1  = torch::where(denom1 == 0, torch::ones_like(denom1), denom1);
    auto term1  = H_km1.slice(2, 0, R) / safe1;
    term1 = torch::where(denom1 == 0, torch::zeros_like(term1), term1);

    // term2[r] = H_km1[r+1] / (t_{r+k+1} - t_{r+1})  -> [B, I, R]
    auto t_r1  = g.slice(2, 1, R + 1);       // [1, I, R]
    auto t_rk1 = g.slice(2, k + 1, R + k + 1); // [1, I, R]
    auto denom2 = (t_rk1 - t_r1);
    auto safe2  = torch::where(denom2 == 0, torch::ones_like(denom2), denom2);
    auto term2  = H_km1.slice(2, 1, R + 1) / safe2;
    term2 = torch::where(denom2 == 0, torch::zeros_like(term2), term2);

    return (term1.scalar_type() == torch::kFloat64
                ? (static_cast<double>(k) * (term1 - term2))
                : (static_cast<float>(k)  * (term1 - term2))
            ).contiguous();
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------
std::pair<torch::Tensor, torch::Tensor> kan_forward_cpu(
    torch::Tensor x,
    torch::Tensor grid,
    torch::Tensor base_weight,
    torch::Tensor spline_weight,
    torch::Tensor spline_scaler,
    int64_t spline_order,
    bool has_scaler
) {
    x              = x.contiguous();
    grid           = grid.contiguous();
    base_weight    = base_weight.contiguous();
    spline_weight  = spline_weight.contiguous();
    if (has_scaler) spline_scaler = spline_scaler.contiguous();

    TORCH_CHECK(x.device().is_cpu(),      "kan_forward_cpu: x must be on CPU");
    kan_check_inputs(x, grid, base_weight, spline_weight, spline_scaler,
                     spline_order, has_scaler);

    const int64_t B = x.size(0);
    const int64_t I = x.size(1);
    const int64_t O = spline_weight.size(0);
    const int64_t R = spline_weight.size(2);

    // --- B-spline basis ---
    auto H = bspline_basis(x, grid, spline_order); // [B, I, R]
    TORCH_CHECK(H.size(2) == R,
        "basis R=", H.size(2), " != spline_weight R=", R);

    // --- effective_weight = spline_weight * scaler (broadcast over R) ---
    torch::Tensor eff_weight;
    if (has_scaler) {
        // spline_scaler: [O, I], spline_weight: [O, I, R]
        eff_weight = spline_weight * spline_scaler.unsqueeze(2);
    } else {
        eff_weight = spline_weight;
    }

    // --- Y_spline = H.view(B, I*R) @ eff_weight.view(O, I*R).T  -> [B, O] ---
    auto H_flat   = H.reshape({B, I * R});                 // [B, I*R]
    auto W_flat   = eff_weight.reshape({O, I * R});        // [O, I*R]
    auto Y_spline = torch::mm(H_flat, W_flat.t());         // [B, O]

    // --- Y_base = silu(x) @ base_weight.T  -> [B, O] ---
    // An empty base_weight means the caller evaluates the base branch.
    if (!kan_has_base(base_weight)) {
        return {Y_spline, H};
    }
    auto silu_x = torch::nn::functional::silu(x);          // [B, I]
    auto Y_base = torch::mm(silu_x, base_weight.t());      // [B, O]

    return {Y_base + Y_spline, H};
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------
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
) {
    grad_output   = grad_output.contiguous();
    x             = x.contiguous();
    grid          = grid.contiguous();
    base_weight   = base_weight.contiguous();
    spline_weight = spline_weight.contiguous();
    basis         = basis.contiguous();
    if (has_scaler) spline_scaler = spline_scaler.contiguous();

    const int64_t B = x.size(0);
    const int64_t I = x.size(1);
    const int64_t O = spline_weight.size(0);
    const int64_t R = spline_weight.size(2);

    // D = grad_output [B, O]
    auto& D = grad_output;
    // H = basis [B, I, R]
    auto& H = basis;

    // --- dW_base[j,i] = sum_b D[b,j] * silu(x[b,i]) ---
    // = D.T @ silu(x)   [O, B] @ [B, I] = [O, I]
    const bool has_base = kan_has_base(base_weight);
    torch::Tensor grad_base;
    if (has_base) {
        auto silu_x = torch::nn::functional::silu(x); // [B, I]
        grad_base   = torch::mm(D.t(), silu_x);       // [O, I]
    }

    // --- effective_weight [O, I, R] ---
    torch::Tensor eff_weight;
    if (has_scaler) {
        eff_weight = spline_weight * spline_scaler.unsqueeze(2);
    } else {
        eff_weight = spline_weight;
    }

    // --- dC[j,i,r] = S[j,i] * sum_b D[b,j]*H[b,i,r] ---
    // sum_b D[b,j]*H[b,i,r] = D.T @ H.reshape(B, I*R) -> [O, I*R] -> reshape [O,I,R]
    auto H_flat = H.reshape({B, I * R});
    auto DtH    = torch::mm(D.t(), H_flat).reshape({O, I, R}); // [O, I, R]
    torch::Tensor grad_spline;
    torch::Tensor grad_scaler;
    if (has_scaler) {
        // dC[j,i,r] = S[j,i] * DtH[j,i,r]
        grad_spline = spline_scaler.unsqueeze(2) * DtH;          // [O, I, R]
        // dS[j,i] = sum_r C[j,i,r] * DtH[j,i,r]  (note: DtH already has the D factor)
        grad_scaler = (spline_weight * DtH).sum(2);              // [O, I]
    } else {
        grad_spline = DtH;                                        // [O, I, R]
        grad_scaler = torch::Tensor(); // undefined
    }

    // --- dX[b,i] contribution from base: W[j,i]*silu'(x[b,i]) ---
    // silu'(x) = sigmoid(x)*(1 + x*(1 - sigmoid(x)))
    torch::Tensor dX_base;
    if (has_base) {
        auto sig_x     = torch::sigmoid(x);                          // [B, I]
        auto silu_grad = sig_x * (1.0 + x * (1.0 - sig_x));         // [B, I]

        // dX_base[b,i] = sum_j D[b,j] * base_weight[j,i] * silu_grad[b,i]
        // = (D @ base_weight)[b,i] * silu_grad[b,i]
        auto Dbase = torch::mm(D, base_weight);    // [B, I]
        dX_base    = Dbase * silu_grad;            // [B, I]
    }

    // --- dX[b,i] contribution from spline: S[j,i] * sum_r C[j,i,r] * dH[b,i,r]/dx ---
    // dH/dx: [B, I, R]
    auto dH_dx = bspline_basis_derivative(x, grid, spline_order); // [B, I, R]

    // Contract: dX_spline[b,i] = sum_j D[b,j] * sum_r eff_weight[j,i,r] * dH_dx[b,i,r]
    // = sum_r dH_dx[b,i,r] * (sum_j D[b,j] * eff_weight[j,i,r])
    // Let A[j,i,r] = eff_weight[j,i,r]  [O, I, R]
    // sum_j D[b,j]*A[j,i,r] = (D.T @ D).unsqueeze ... no
    // Better: einsum "bj,jir->bir" then sum over r
    // = D @ eff_weight.reshape(O, I*R) -> [B, I*R] -> reshape [B, I, R]
    auto DeffW = torch::mm(D, eff_weight.reshape({O, I * R}))
                     .reshape({B, I, R}); // [B, I, R]
    auto dX_spline = (DeffW * dH_dx).sum(2); // [B, I]

    auto grad_x = has_base ? dX_base + dX_spline : dX_spline;

    if (has_scaler) {
        return {grad_x, grad_base, grad_spline, grad_scaler};
    } else {
        return {grad_x, grad_base, grad_spline, torch::Tensor()};
    }
}

// ---------------------------------------------------------------------------
// Composite (differentiable) backward
//
// Same maths as kan_backward_cpu, but the basis is recomputed from x with
// differentiable ATen ops instead of being taken from the saved forward
// buffer / native derivative kernel.  When autograd records this backward
// (create_graph=True), d(grad)/dx therefore includes the spline curvature and
// higher-order derivatives are exact.  Works for CPU and CUDA tensors.
// ---------------------------------------------------------------------------
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
) {
    const int64_t B = x.size(0);
    const int64_t I = x.size(1);
    const int64_t O = spline_weight.size(0);
    const int64_t R = spline_weight.size(2);
    const bool has_base = kan_has_base(base_weight);
    auto& D = grad_output;

    auto H = bspline_basis(x, grid, spline_order);            // [B, I, R]
    auto eff_weight = has_scaler ? spline_weight * spline_scaler.unsqueeze(2)
                                 : spline_weight;

    auto DtH = torch::mm(D.t(), H.reshape({B, I * R})).reshape({O, I, R});
    torch::Tensor grad_spline, grad_scaler;
    if (has_scaler) {
        grad_spline = spline_scaler.unsqueeze(2) * DtH;
        grad_scaler = (spline_weight * DtH).sum(2);
    } else {
        grad_spline = DtH;
    }

    auto dH_dx = bspline_basis_derivative(x, grid, spline_order);
    auto DeffW = torch::mm(D, eff_weight.reshape({O, I * R})).reshape({B, I, R});
    auto grad_x = (DeffW * dH_dx).sum(2);

    torch::Tensor grad_base;
    if (has_base) {
        auto silu_x = torch::silu(x);
        grad_base = torch::mm(D.t(), silu_x);
        auto sig_x = torch::sigmoid(x);
        auto silu_grad = sig_x * (1.0 + x * (1.0 - sig_x));
        grad_x = grad_x + torch::mm(D, base_weight) * silu_grad;
    }
    return {grad_x, grad_base, grad_spline, grad_scaler};
}
