#!/usr/bin/env python3
"""
Example: Training a KAN on a simple regression task.
Demonstrates: CPU/CUDA usage, grid update, optimizer state handling,
state_dict save/load, regularization, higher-order derivatives.
"""
import sys
import os
sys.path.insert(0, "/home/hy/torch-spline-kan")
import torch
import torch.nn as nn

from spline_kan import KAN, KANLinear


def example_basic_training():
    print("=== Basic Training Example ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = KAN([4, 16, 1], grid_size=5, spline_order=3).to(device)
    print(f"  Backend selected: {model.layers[0].selected_backend}")

    x = torch.randn(128, 4, device=device)
    y_target = torch.sin(x[:, :1]) + torch.cos(x[:, 1:2])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(50):
        optimizer.zero_grad()
        y = model(x)
        loss = nn.MSELoss()(y, y_target)
        # Activation-mode regularization: pass x so each layer evaluates its
        # actual input distribution.  Recreate the optimizer after update_grid.
        reg = model.regularization_loss(0.01, 0.01, x=x)
        (loss + reg).backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")

    print("  Training done.")


def example_grid_update_and_optimizer():
    print("\n=== Grid Update + Optimizer State ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KAN([4, 8, 1], grid_size=5, spline_order=3).to(device)

    x = torch.randn(200, 4, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(10):
        optimizer.zero_grad()
        y = model(x)
        y.sum().backward()
        optimizer.step()

    print("  Updating grid (adapts knots to data distribution)...")
    with torch.no_grad():
        model.update_grid(x)

    # Recreate optimizer to clear stale momentum after grid update.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print("  Optimizer recreated after grid update (clears stale momentum).")

    for _ in range(10):
        optimizer.zero_grad()
        y = model(x)
        y.sum().backward()
        optimizer.step()
    print("  Post-update training OK.")


def example_save_load():
    print("\n=== Save/Load State Dict ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KAN([4, 8, 2], grid_size=5, spline_order=3).to(device)

    x = torch.randn(20, 4, device=device)
    y_before = model(x)

    torch.save(model.state_dict(), "/tmp/kan_test.pt")
    model2 = KAN([4, 8, 2], grid_size=5, spline_order=3).to(device)
    model2.load_state_dict(torch.load("/tmp/kan_test.pt", map_location=device))

    y_after = model2(x)
    assert torch.allclose(y_before, y_after, atol=1e-6), "Save/load mismatch!"
    print("  Save/load round-trip: PASSED")


def example_pde_higher_order():
    print("\n=== Higher-Order Differentiation (PDE) ===")
    # Solve Poisson PDE: u_xx = -pi^2 * sin(pi*x)
    # True solution: u(x) = sin(pi*x)
    device = "cpu"
    model = KAN([1, 16, 1], grid_size=10, spline_order=3, backend="reference").double().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.linspace(-0.9, 0.9, 30, dtype=torch.float64, device=device, requires_grad=True).unsqueeze(1)

    for step in range(30):
        optimizer.zero_grad()
        u = model(x)
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
        rhs = -(torch.pi**2) * torch.sin(torch.pi * x)
        pde_loss = ((u_xx - rhs)**2).mean()
        pde_loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"  Step {step}: PDE residual={pde_loss.item():.6f}")

    print("  PDE training done (higher-order grad through reference backend).")


if __name__ == "__main__":
    example_basic_training()
    example_grid_update_and_optimizer()
    example_save_load()
    example_pde_higher_order()
    print("\nAll examples completed.")
