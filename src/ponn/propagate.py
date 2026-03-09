# src/ponn/propagate.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .dynamics import SSDynamicsCoeffs, ss_rhs_state as rhs_state


@dataclass(frozen=True)
class PropagateResult:
    t: torch.Tensor      # (N,)
    x: torch.Tensor      # (N,6)  [r,v]
    r: torch.Tensor      # (N,3)
    v: torch.Tensor      # (N,3)


def lerp_u(t: torch.Tensor, t_grid: torch.Tensor, u_grid: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation of u(t) given samples on (t_grid, u_grid).
    t: (...,)
    t_grid: (N,)
    u_grid: (N,3)
    returns u: (...,3)
    """
    if t_grid.dim() != 1:
        raise ValueError("t_grid must be 1D")
    if u_grid.shape != (t_grid.numel(), 3):
        raise ValueError("u_grid must have shape (N,3)")

    # clamp to domain
    t0 = t_grid[0]
    tf = t_grid[-1]
    tt = t.clamp(min=t0, max=tf)

    # find right index j s.t. t_grid[j] <= t < t_grid[j+1]
    j = torch.searchsorted(t_grid, tt, right=True) - 1
    j = j.clamp(0, t_grid.numel() - 2)

    t_left = t_grid[j]
    t_right = t_grid[j + 1]
    u_left = u_grid[j]
    u_right = u_grid[j + 1]

    denom = (t_right - t_left).clamp_min(1e-15)
    alpha = ((tt - t_left) / denom).unsqueeze(-1)  # (...,1)
    return (1 - alpha) * u_left + alpha * u_right


def hold_u(t: torch.Tensor, t_grid: torch.Tensor, u_grid: torch.Tensor) -> torch.Tensor:
    """
    Piecewise-constant (zero-order hold) sampling of u(t).
    On interval [t_grid[j], t_grid[j+1]) use u_grid[j].
    t: (...,)
    t_grid: (N,)
    u_grid: (N,3)
    returns u: (...,3)
    """
    if t_grid.dim() != 1:
        raise ValueError("t_grid must be 1D")
    if u_grid.shape != (t_grid.numel(), 3):
        raise ValueError("u_grid must have shape (N,3)")

    # clamp to domain
    t0 = t_grid[0]
    tf = t_grid[-1]
    tt = t.clamp(min=t0, max=tf)

    # j such that t_grid[j] <= t < t_grid[j+1]
    j = torch.searchsorted(t_grid, tt, right=True) - 1
    j = j.clamp(0, t_grid.numel() - 1)  # allow last point -> last control

    # if tt == tf, searchsorted gives N, so j becomes N-1 -> ok
    return u_grid[j]


def _rk4_step(
    x: torch.Tensor,  # (6,)
    t: torch.Tensor,  # scalar tensor
    dt: torch.Tensor, # scalar tensor
    t_grid: torch.Tensor,
    u_grid: torch.Tensor,
    coeffs: SSDynamicsCoeffs,
) -> torch.Tensor:
    """
    One RK4 step for xdot = f(x,t) with u(t) from linear interpolation on u_grid.
    """
    def f(x_, t_):
        u_ = lerp_u(t_, t_grid, u_grid).reshape(3)  # ✅ use public function
        return rhs_state(x_.reshape(1, 6), u_.reshape(1, 3), coeffs).reshape(6)

    k1 = f(x, t)
    k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(x + dt * k3, t + dt)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def propagate_ss_rv(
    t_grid: torch.Tensor,   # (N,) increasing
    r0: torch.Tensor,       # (3,)
    v0: torch.Tensor,       # (3,)
    u_grid: torch.Tensor,   # (N,3) control samples aligned with t_grid
    coeffs: SSDynamicsCoeffs,
) -> PropagateResult:
    """
    Forward integrate SS dynamics using RK4 on the same grid as the control samples.
    Control is linearly interpolated inside each RK4 evaluation.
    Returns states at the same t_grid points.
    """
    if t_grid.dim() != 1:
        raise ValueError("t_grid must be (N,)")
    N = t_grid.numel()
    if u_grid.shape != (N, 3):
        raise ValueError("u_grid must be (N,3)")
    if r0.shape != (3,) or v0.shape != (3,):
        raise ValueError("r0,v0 must be (3,)")

    device = t_grid.device
    dtype = t_grid.dtype

    x = torch.zeros((N, 6), device=device, dtype=dtype)
    x0 = torch.cat([r0, v0], dim=0)
    x[0] = x0

    for i in range(N - 1):
        t = t_grid[i]
        dt = t_grid[i + 1] - t_grid[i]
        x[i + 1] = _rk4_step(x[i], t, dt, t_grid, u_grid, coeffs)

    r = x[:, 0:3]
    v = x[:, 3:6]
    return PropagateResult(t=t_grid, x=x, r=r, v=v)