# src/ponn/losses.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .dynamics import SSDynamicsCoeffs
from .ponn import control_from_costate, lam_r_dot_rhs, lam_v_dot_rhs
from .constraint import ConeConstraint, ControlConstraint, C_a, C_gs_from_cone


@dataclass(frozen=True)
class RendezvousLossWeights:
    w_v: float = 3000.0
    w_lam_r: float = 1.0
    w_lam_v: float = 1.0
    w_gs: float = 700.0
    w_a: float = 700.0


def fb_phi(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    if not torch.is_tensor(b):
        b = torch.tensor(float(b), device=a.device, dtype=a.dtype)
    else:
        b = b.to(device=a.device, dtype=a.dtype)
    return torch.sqrt(a * a + b * b + eps) + a - b


@dataclass
class RendezvousLosses:
    """
    PoNN loss pack for the paper's Energy-Optimal Rendezvous example:
      Lv, Lλr, Lλv, Lgs(FB), La(FB)
    """
    coeffs: SSDynamicsCoeffs
    weights: RendezvousLossWeights
    cone_cstr: ConeConstraint
    ctrl_cstr: ControlConstraint
    t0: float
    tf: float
    near_tol_km: float = 1e-3   # moved here (was referenced before)

    def residual_Lv(
        self,
        *,
        r: torch.Tensor, v: torch.Tensor, vdot: torch.Tensor,
        lam_v: torch.Tensor, mu_a: torch.Tensor,
    ) -> torch.Tensor:
        ctrl = control_from_costate(lam_v, mu_a, ctrl=self.ctrl_cstr)
        u = ctrl.u
        vdot_rhs = (r @ self.coeffs.M.T) + (v @ self.coeffs.N.T) + u
        return float(self.weights.w_v) * (vdot - vdot_rhs)

    def residual_Llam_r(
        self,
        *,
        r: torch.Tensor,
        lam_v: torch.Tensor,
        lam_r_dot: torch.Tensor,
        mu_gs: torch.Tensor,
    ) -> torch.Tensor:
        
        if r.ndim != 2 or lam_v.ndim != 2 or lam_r_dot.ndim != 2 or r.shape[-1] != 3 or lam_v.shape[-1] != 3 or lam_r_dot.shape[-1] != 3:
            raise ValueError("r, lam_v, lam_r_dot must be (N,3)")
        
        N = r.shape[0]
        t = torch.linspace(float(self.t0), float(self.tf), N, device=r.device, dtype=r.dtype)

        rhs = lam_r_dot_rhs(
            r=r,
            t=t,
            lam_v=lam_v,
            mu_gs=mu_gs,
            coeffs=self.coeffs,
            cone=self.cone_cstr,
            near_tol_km=float(self.near_tol_km),
        )
        return float(self.weights.w_lam_r) * (lam_r_dot - rhs)

    def residual_Llam_v(
        self,
        *,
        lam_r: torch.Tensor, lam_v: torch.Tensor, lam_v_dot: torch.Tensor,
    ) -> torch.Tensor:
        rhs = lam_v_dot_rhs(lam_r=lam_r, lam_v=lam_v, coeffs=self.coeffs)
        return float(self.weights.w_lam_v) * (lam_v_dot - rhs)

    def residual_Lgs(
        self,
        *,
        r: torch.Tensor,
        mu_gs: torch.Tensor,
    ) -> torch.Tensor:
        
        if r.ndim != 2 or r.shape[-1] != 3:
            raise ValueError("r must be (N,3)")
        
        N = r.shape[0]
        t = torch.linspace(float(self.t0), float(self.tf), N, device=r.device, dtype=r.dtype)

        C = C_gs_from_cone(r, cone=self.cone_cstr, t=t)

        # avoid singularity at r≈rf (optional; keep your current behavior)
        r_rel_norm = torch.linalg.norm(r - self.cone_cstr.rf, dim=-1)
        near = r_rel_norm <= float(self.near_tol_km)
        C = torch.where(near, torch.zeros_like(C), C)

        if not torch.is_tensor(mu_gs):
            mu = torch.tensor(float(mu_gs), device=C.device, dtype=C.dtype)
            mu = mu.expand_as(C)
        else:
            mu = mu_gs.to(device=C.device, dtype=C.dtype)
            if mu.ndim == 0:
                mu = mu.expand_as(C)
            elif mu.shape != C.shape:
                raise ValueError(f"mu_gs shape {tuple(mu.shape)} must match C {tuple(C.shape)} or be scalar")
        
        return float(self.weights.w_gs) * fb_phi(C, mu, eps=float(self.cone_cstr.eps))

    def residual_La(
        self,
        *,
        lam_v: torch.Tensor,
        mu_a: torch.Tensor,
    ) -> torch.Tensor:
        ctrl = control_from_costate(lam_v, mu_a, ctrl=self.ctrl_cstr)
        a_c = ctrl.a_c
        C = C_a(a_c, float(self.ctrl_cstr.a_c_max))

        if not torch.is_tensor(mu_a):
            mu = torch.tensor(float(mu_a), device=C.device, dtype=C.dtype)
        else:
            mu = mu_a.to(device=C.device, dtype=C.dtype)

        return float(self.weights.w_a) * fb_phi(C, mu, eps=float(self.ctrl_cstr.eps))

    def compute(
        self,
        *,
        r: torch.Tensor, v: torch.Tensor, vdot: torch.Tensor,
        lam_r: torch.Tensor, lam_r_dot: torch.Tensor,
        lam_v: torch.Tensor, lam_v_dot: torch.Tensor,
        mu_a: torch.Tensor | float,
        mu_gs: torch.Tensor | float,
    ) -> Dict[str, torch.Tensor]:
        return {
            "Lv": self.residual_Lv(r=r, v=v, vdot=vdot, lam_v=lam_v, mu_a=mu_a),
            "Llam_r": self.residual_Llam_r(r=r, lam_v=lam_v, lam_r_dot=lam_r_dot, mu_gs=mu_gs),
            "Llam_v": self.residual_Llam_v(lam_r=lam_r, lam_v=lam_v, lam_v_dot=lam_v_dot),
            "Lgs": self.residual_Lgs(r=r, mu_gs=mu_gs),
            "La": self.residual_La(lam_v=lam_v, mu_a=mu_a),
        }

    def pack(self, blocks: Dict[str, torch.Tensor]) -> torch.Tensor:
        Lv = blocks["Lv"].reshape(-1)
        Llam_r = blocks["Llam_r"].reshape(-1)
        Llam_v = blocks["Llam_v"].reshape(-1)
        Lgs = blocks["Lgs"].reshape(-1)
        La = blocks["La"].reshape(-1)
        return torch.cat([Lv, Llam_r, Llam_v, Lgs, La], dim=0)