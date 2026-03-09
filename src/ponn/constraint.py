# src/ponn/constraint.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch


@dataclass(frozen=True)
class ConeActivation:
    """
    Cone activation parameters (Eq.38/39).
    mode='time' uses (t_c, k)
    mode='dist' uses (r_c, k)
    """
    mode: Literal["time", "dist"] = "time"
    t_c: Optional[float] = None
    r_c: Optional[float] = None
    k: float = 1000.0


@dataclass(frozen=True)
class ConeConstraint:
    """
    Cone (glideslope) constraint parameters.
    """
    rf: torch.Tensor          # (3,)
    n_hat: torch.Tensor       # (3,)
    gamma_max_deg: float
    activation: ConeActivation
    eps: float = 1e-12

    def to(self, *, device=None, dtype=None) -> "ConeConstraint":
        return ConeConstraint(
            rf=self.rf.to(device=device, dtype=dtype),
            n_hat=self.n_hat.to(device=device, dtype=dtype),
            gamma_max_deg=float(self.gamma_max_deg),
            activation=self.activation,
            eps=float(self.eps),
        )


@dataclass(frozen=True)
class ControlConstraint:
    """
    Control magnitude constraint parameters.
    """
    a_c_max: float
    eps: float = 1e-12


# ----------------------------
# Basic constraints
# ----------------------------

def C_a(a_c: torch.Tensor, a_c_max: float | torch.Tensor) -> torch.Tensor:
    """
    Eq.(35): C_a = a_c - a_c,max <= 0
    """
    if not torch.is_tensor(a_c_max):
        a_c_max = torch.tensor(float(a_c_max), device=a_c.device, dtype=a_c.dtype)
    else:
        a_c_max = a_c_max.to(device=a_c.device, dtype=a_c.dtype)
    return a_c - a_c_max


def C_a_from_ctrl(a_c: torch.Tensor, ctrl: ControlConstraint) -> torch.Tensor:
    """
    Convenience wrapper using ControlConstraint dataclass.
    """
    return C_a(a_c, ctrl.a_c_max)


# ----------------------------
# Activation k_act (Eq.38 / Eq.39)
# ----------------------------

def k_act_time(t: torch.Tensor, t_c: float | torch.Tensor, k: float | torch.Tensor) -> torch.Tensor:
    """
    Eq.(39): k_act(t) = 1/2 + 1/2 * tanh( k (t - t_c) )
    """
    if not torch.is_tensor(t_c):
        t_c = torch.tensor(float(t_c), device=t.device, dtype=t.dtype)
    else:
        t_c = t_c.to(device=t.device, dtype=t.dtype)

    if not torch.is_tensor(k):
        k = torch.tensor(float(k), device=t.device, dtype=t.dtype)
    else:
        k = k.to(device=t.device, dtype=t.dtype)

    return 0.5 + 0.5 * torch.tanh(k * (t - t_c))


def k_act_dist(r_rel_norm: torch.Tensor, r_c: float | torch.Tensor, k: float | torch.Tensor) -> torch.Tensor:
    """
    Eq.(38): k_act(r) = 1/2 + 1/2 * tanh( k (r_rel_norm - r_c) )
    """
    if not torch.is_tensor(r_c):
        r_c = torch.tensor(float(r_c), device=r_rel_norm.device, dtype=r_rel_norm.dtype)
    else:
        r_c = r_c.to(device=r_rel_norm.device, dtype=r_rel_norm.dtype)

    if not torch.is_tensor(k):
        k = torch.tensor(float(k), device=r_rel_norm.device, dtype=r_rel_norm.dtype)
    else:
        k = k.to(device=r_rel_norm.device, dtype=r_rel_norm.dtype)

    return 0.5 + 0.5 * torch.tanh(k * (r_rel_norm - r_c))


# ----------------------------
# Cone constraint C_gs (Eq.36-37) and gradient (Eq.46)
# ----------------------------

def C_gs_gen(
    r: torch.Tensor,
    rf: torch.Tensor,
    n_hat: torch.Tensor,
    gamma_max_deg: float,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Eq.(36): C_gs,gen = cos(gamma_gs,max) - r_hat_rel · n_hat <= 0
    """
    if r.shape[-1] != 3:
        raise ValueError("r must have last dim 3")
    if rf.shape[-1] != 3 or n_hat.shape[-1] != 3:
        raise ValueError("rf and n_hat must have last dim 3")

    r_rel = r - rf
    r_rel_norm = torch.linalg.norm(r_rel, dim=-1).clamp_min(eps)
    r_hat_rel = r_rel / r_rel_norm.unsqueeze(-1)

    n_norm = torch.linalg.norm(n_hat, dim=-1).clamp_min(eps)
    n_unit = n_hat / n_norm.unsqueeze(-1)

    cos_g = torch.cos(torch.tensor(float(gamma_max_deg) * torch.pi / 180.0,
                                   device=r.device, dtype=r.dtype))
    dot = (r_hat_rel * n_unit).sum(dim=-1)
    return cos_g - dot


def C_gs(
    r: torch.Tensor,
    rf: torch.Tensor,
    n_hat: torch.Tensor,
    gamma_max_deg: float,
    *,
    mode: Literal["time", "dist"] = "time",
    t: Optional[torch.Tensor] = None,
    t_c: Optional[float] = None,
    r_c: Optional[float] = None,
    k: float = 1000.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Eq.(37): C_gs = k_act * C_gs,gen
    """
    c_gen = C_gs_gen(r, rf, n_hat, gamma_max_deg, eps=eps)

    if mode == "time":
        if t is None or t_c is None:
            raise ValueError("mode='time' requires t and t_c")
        kact = k_act_time(t, t_c, k)
    elif mode == "dist":
        if r_c is None:
            raise ValueError("mode='dist' requires r_c")
        r_rel = r - rf
        r_rel_norm = torch.linalg.norm(r_rel, dim=-1).clamp_min(eps)
        kact = k_act_dist(r_rel_norm, r_c, k)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return kact * c_gen


def pC_gs_pr(
    r: torch.Tensor,
    rf: torch.Tensor,
    n_hat: torch.Tensor,
    gamma_max_deg: float,  # unused
    *,
    mode: Literal["time", "dist"] = "time",
    t: Optional[torch.Tensor] = None,
    t_c: Optional[float] = None,
    r_c: Optional[float] = None,
    k: float = 1000.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Eq.(46): ∂C_gs/∂r = k_act * ∂/∂r (cos(gamma_max) - r_hat_rel · n_hat)
    """
    if r.shape[-1] != 3:
        raise ValueError("r must have last dim 3")
    if rf.shape[-1] != 3 or n_hat.shape[-1] != 3:
        raise ValueError("rf and n_hat must have last dim 3")

    r_rel = r - rf
    r_norm = torch.linalg.norm(r_rel, dim=-1).clamp_min(eps)
    r_norm3 = (r_norm * r_norm * r_norm).clamp_min(eps)

    n_norm = torch.linalg.norm(n_hat, dim=-1).clamp_min(eps)
    n_unit = n_hat / n_norm.unsqueeze(-1)

    ndotr = (n_unit * r_rel).sum(dim=-1)

    term1 = -n_unit / r_norm.unsqueeze(-1)
    term2 = (ndotr.unsqueeze(-1) * r_rel) / r_norm3.unsqueeze(-1)
    grad_gen = term1 + term2

    if mode == "time":
        if t is None or t_c is None:
            raise ValueError("mode='time' requires t and t_c")
        kact = k_act_time(t, t_c, k)
    elif mode == "dist":
        if r_c is None:
            raise ValueError("mode='dist' requires r_c")
        kact = k_act_dist(r_norm, r_c, k)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return kact.unsqueeze(-1) * grad_gen


# ----------------------------
# Dataclass-based wrappers (recommended API)
# ----------------------------

def C_gs_from_cone(
    r: torch.Tensor,
    *,
    cone: ConeConstraint,
    t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Wrapper that uses ConeConstraint + ConeActivation dataclasses.
    """
    act = cone.activation
    if act.mode == "time":
        if t is None:
            raise ValueError("C_gs_from_cone(mode='time') requires t")
        if act.t_c is None:
            raise ValueError("ConeActivation.t_c is required for mode='time'")
        return C_gs(
            r=r,
            rf=cone.rf,
            n_hat=cone.n_hat,
            gamma_max_deg=cone.gamma_max_deg,
            mode="time",
            t=t,
            t_c=act.t_c,
            k=act.k,
            eps=cone.eps,
        )
    else:
        if act.r_c is None:
            raise ValueError("ConeActivation.r_c is required for mode='dist'")
        return C_gs(
            r=r,
            rf=cone.rf,
            n_hat=cone.n_hat,
            gamma_max_deg=cone.gamma_max_deg,
            mode="dist",
            r_c=act.r_c,
            k=act.k,
            eps=cone.eps,
        )


def pC_gs_pr_from_cone(
    r: torch.Tensor,
    *,
    cone: ConeConstraint,
    t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Wrapper that uses ConeConstraint + ConeActivation dataclasses.
    """
    act = cone.activation
    if act.mode == "time":
        if t is None:
            raise ValueError("pC_gs_pr_from_cone(mode='time') requires t")
        if act.t_c is None:
            raise ValueError("ConeActivation.t_c is required for mode='time'")
        return pC_gs_pr(
            r=r,
            rf=cone.rf,
            n_hat=cone.n_hat,
            gamma_max_deg=cone.gamma_max_deg,
            mode="time",
            t=t,
            t_c=act.t_c,
            k=act.k,
            eps=cone.eps,
        )
    else:
        if act.r_c is None:
            raise ValueError("ConeActivation.r_c is required for mode='dist'")
        return pC_gs_pr(
            r=r,
            rf=cone.rf,
            n_hat=cone.n_hat,
            gamma_max_deg=cone.gamma_max_deg,
            mode="dist",
            r_c=act.r_c,
            k=act.k,
            eps=cone.eps,
        )