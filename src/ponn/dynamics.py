# src/ponn/dynamics.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch


@dataclass(frozen=True)
class SSOrbitParams:
    Re_km: float
    Rt_km: float
    it_rad: float
    mu_km3_s2: float
    J2: float


@dataclass(frozen=True)
class SSDynamicsCoeffs:
    omega: float   # [1/s]
    K: float       # [-]
    e: float       # [1/s]
    f: float       # [1/s]
    M: torch.Tensor  # (3,3) [1/s^2]
    N: torch.Tensor  # (3,3) [1/s]


def compute_ss_coeffs(
    orbit: SSOrbitParams,
    *,
    device=None,
    dtype: Optional[torch.dtype] = None,
) -> SSDynamicsCoeffs:
    """
    Paper Eq.(33):
      ω = sqrt(mu / Rt^3)
      K = (3 J2 Re^2)/(8 Rt^2) * (1 + 3 cos(2 i_t))
      f = ω sqrt(1 - K)
      e = ω sqrt(1 + K)

      M = [[4e^2 - f^2, 0, 0],
           [0, 0, 0],
           [0, 0, -(2e^2 - f^2)]]

      N = [[0, 2e, 0],
           [-2e, 0, 0],
           [0, 0, 0]]
    """
    if orbit.Rt_km <= 0 or orbit.Re_km <= 0 or orbit.mu_km3_s2 <= 0:
        raise ValueError("Rt_km, Re_km, mu_km3_s2 must be positive")
    if orbit.J2 < 0:
        raise ValueError("J2 must be non-negative")

    # --- scalars (float) ---
    omega = (orbit.mu_km3_s2 / (orbit.Rt_km ** 3)) ** 0.5
    K = (3.0 * orbit.J2 * (orbit.Re_km ** 2)) / (8.0 * (orbit.Rt_km ** 2)) * (
        1.0 + 3.0 * float(torch.cos(torch.tensor(2.0 * orbit.it_rad)))
    )

    # sqrt domain guard (numerical)
    one_minus_K = max(0.0, 1.0 - K)
    one_plus_K  = max(0.0, 1.0 + K)

    f = omega * (one_minus_K ** 0.5)
    e = omega * (one_plus_K ** 0.5)

    # --- matrices ---
    M = torch.tensor(
        [[4.0 * e * e - f * f, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, -(2.0 * e * e - f * f)]],
        device=device,
        dtype=dtype,
    )
    N = torch.tensor(
        [[0.0, 2.0 * e, 0.0],
         [-2.0 * e, 0.0, 0.0],
         [0.0, 0.0, 0.0]],
        device=device,
        dtype=dtype,
    )

    return SSDynamicsCoeffs(omega=omega, K=K, e=e, f=f, M=M, N=N)


def ss_rhs_rv(
    r: torch.Tensor,   # (...,3)
    v: torch.Tensor,   # (...,3)
    u: torch.Tensor,   # (...,3) where u = a_c * a_hat_c
    coeffs: SSDynamicsCoeffs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if r.shape[-1] != 3 or v.shape[-1] != 3 or u.shape[-1] != 3:
        raise ValueError("r, v, u must have last dim 3")

    rdot = v
    # vdot = M r + N v + u  (paper Eq. 32)
    vdot = r @ coeffs.M.T + v @ coeffs.N.T + u
    return rdot, vdot


def ss_rhs_state(
    x: torch.Tensor,   # (...,6) = [r(3), v(3)]
    u: torch.Tensor,   # (...,3)
    coeffs: SSDynamicsCoeffs,
) -> torch.Tensor:
    if x.shape[-1] != 6:
        raise ValueError("state must have last dim 6")
    r = x[..., 0:3]
    v = x[..., 3:6]
    rdot, vdot = ss_rhs_rv(r, v, u, coeffs)
    return torch.cat([rdot, vdot], dim=-1)