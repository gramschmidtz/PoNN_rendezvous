# src/ponn/ponn.py
from __future__ import annotations

from dataclasses import dataclass
import torch

from .dynamics import SSDynamicsCoeffs
from .constraint import ConeConstraint, ControlConstraint, pC_gs_pr_from_cone


@dataclass(frozen=True)
class ControlFromCostate:
    a_c_hat: torch.Tensor   # (N,3)
    a_c: torch.Tensor       # (N,)  magnitude
    u: torch.Tensor         # (N,3) = a_c * a_hat
    lam_norm: torch.Tensor  # (N,)  ||lambda_v||


def control_from_costate(
    lambda_v: torch.Tensor,           # (N,3)
    mu_a: torch.Tensor | float,       # scalar or (N,)
    *,
    ctrl: ControlConstraint | None = None,
    eps: float | None = None,
) -> ControlFromCostate:
    """
    Implements:
      a_c_hat = -lambda_v / ||lambda_v||
      a_c     = ||lambda_v|| - mu_a
      u       = a_c * a_c_hat

    eps priority:
      1) explicit eps
      2) ctrl.eps if ctrl is provided
      3) default 1e-12
    """
    if lambda_v.shape[-1] != 3:
        raise ValueError("lambda_v must have last dim 3")

    if eps is None:
        eps = float(ctrl.eps) if ctrl is not None else 1e-12

    # ensure mu_a tensor
    if not torch.is_tensor(mu_a):
        mu_a = torch.tensor(float(mu_a), device=lambda_v.device, dtype=lambda_v.dtype)
    else:
        mu_a = mu_a.to(device=lambda_v.device, dtype=lambda_v.dtype)

    lam_norm = torch.linalg.norm(lambda_v, dim=-1)  # (...,)

    # broadcast scalar -> (...,)
    if mu_a.ndim == 0:
        mu_a = mu_a.expand_as(lam_norm)
    else:
        if mu_a.shape != lam_norm.shape:
            raise ValueError(f"mu_a shape {tuple(mu_a.shape)} must match lam_norm {tuple(lam_norm.shape)} or be scalar")

    denom = lam_norm.clamp_min(eps).unsqueeze(-1)  # (...,1)
    a_c_hat = -lambda_v / denom
    a_c = (lam_norm - mu_a).clamp(min=0.0)
    u = a_c_hat * a_c.unsqueeze(-1)
    return ControlFromCostate(a_c_hat=a_c_hat, a_c=a_c, u=u, lam_norm=lam_norm)


def lam_r_dot_rhs(
    *,
    r: torch.Tensor,               # (...,3)
    t: torch.Tensor,               # (...,)
    lam_v: torch.Tensor,           # (...,3)
    mu_gs: torch.Tensor,           # (...,)
    coeffs: SSDynamicsCoeffs,
    cone: ConeConstraint,
    near_tol_km: float = 1e-2,
) -> torch.Tensor:
    r"""
    Implements:
      \dot{λ}_r = -M^T λ_v - μ_gs * ∂C_gs/∂r
    Using row-vector convention: -M^T λ_v == -(λ_v @ M)
    """
    if r.shape[-1] != 3 or lam_v.shape[-1] != 3:
        raise ValueError("r and lam_v must have last dim 3")

    # ensure mu_gs tensor scalar
    if not torch.is_tensor(mu_gs):
        mu_gs = torch.tensor(float(mu_gs), device=r.device, dtype=r.dtype)
    else:
        mu_gs = mu_gs.to(device=r.device, dtype=r.dtype)
    
    # broadcast scalar -> (...,)
    if mu_gs.ndim == 0:
        mu_gs = mu_gs.expand(r.shape[:-1])
    else:
        if mu_gs.shape != r.shape[:-1]:
            raise ValueError(f"mu_gs shape {tuple(mu_gs.shape)} must match r {tuple(r.shape[:-1])} or be scalar")

    # -M^T lam_v  (row form)
    term_M = -(lam_v @ coeffs.M)  # (...,3)

    # cone gradient term (dataclass-based)
    pC_gs_pr = pC_gs_pr_from_cone(r, cone=cone, t=t)  # (...,3)

    # near-target: disable cone gradient to avoid singularity
    r_rel_norm = torch.linalg.norm(r - cone.rf, dim=-1)
    near = r_rel_norm <= float(near_tol_km)
    pC_gs_pr = torch.where(near.unsqueeze(-1), torch.zeros_like(pC_gs_pr), pC_gs_pr)

    return term_M - mu_gs.unsqueeze(-1) * pC_gs_pr


def lam_v_dot_rhs(
    *,
    lam_r: torch.Tensor,     # (...,3)
    lam_v: torch.Tensor,     # (...,3)
    coeffs: SSDynamicsCoeffs,
) -> torch.Tensor:
    r"""
    Implements:
      \dot{λ}_v = -λ_r - N^T λ_v
    Using row-vector convention: -N^T λ_v == -(λ_v @ N)
    """
    if lam_r.shape[-1] != 3 or lam_v.shape[-1] != 3:
        raise ValueError("lam_r and lam_v must have last dim 3")
    return -lam_r - (lam_v @ coeffs.N)