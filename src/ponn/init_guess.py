# src/ponn/init_guess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Dict, Any

import sys
import torch
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .ce import CEModel, CEBetas, omegas_A6
from .dynamics import SSDynamicsCoeffs
from .constraint import pC_gs_pr_from_cone, ConeConstraint, ConeActivation


@dataclass(frozen=True)
class TrainParams:
    betas: CEBetas
    mu_gs: torch.Tensor  # (N,)
    mu_a: torch.Tensor   # (N,)

    def to(self, *, device=None, dtype=None) -> "TrainParams":
        device = device if device is not None else self.mu_a.device
        dtype = dtype if dtype is not None else self.mu_a.dtype
        return TrainParams(
            betas=self.betas.to(device=device, dtype=dtype),
            mu_a=self.mu_a.to(device=device, dtype=dtype),
            mu_gs=self.mu_gs.to(device=device, dtype=dtype),
        )

    def pack(self) -> torch.Tensor:

        if self.mu_a.ndim != 1:
            raise ValueError(f"mu_a must be 1D (N,), got {tuple(self.mu_a.shape)}")
        if self.mu_gs.ndim != 1:
            raise ValueError(f"mu_gs must be 1D (N,), got {tuple(self.mu_a.shape)}")
        
        L = self.betas.beta_r.shape[-1]

        if self.betas.beta_r.shape != (3,L):
            raise ValueError(f"beta_r must be (3,L), got {tuple(self.betas.beta_r.shape)}")
        if self.betas.beta_lam_r.shape != (3,L):
            raise ValueError(f"beta_lam_r must be (3,L), got {tuple(self.betas.beta_lam_r.shape)}")
        if self.betas.beta_lam_v.shape != (3,L):
            raise ValueError(f"beta_lam_v must be (3,L), got {tuple(self.betas.beta_lam_v.shape)}")
        
        N = self.mu_a.numel()

        if self.mu_a.numel() != N:
            raise ValueError(f"mu_gs length {self.mu_gs.numel()} must match mu_a length {N}")
        
        return torch.cat(
            [
                self.betas.beta_r.reshape(-1),
                self.betas.beta_lam_r.reshape(-1),
                self.betas.beta_lam_v.reshape(-1),
                self.mu_gs.reshape(-1),
                self.mu_a.reshape(-1),
            ],
            dim=0
        )

    @classmethod
    def unpack(cls, train_vec: torch.Tensor, L: int, N: int, *, device=None, dtype=None) -> "TrainParams":
        
        if not torch.is_tensor(train_vec):
            raise TypeError(f"train_vec must be torch.Tensor, got {type(train_vec)}")
        if train_vec.ndim != 1:
            raise ValueError(f"train_vec must be 1D, got shape={tuple(train_vec.shape)}")
        
        if device is not None or dtype is not None:
            train_vec = train_vec.to(device=device, dtype=dtype)

        expected = 9 * L + 2 * N
        if train_vec.numel() != expected:
            raise ValueError(f"train_vec.numel()={train_vec.numel()} != expected={expected} (L={L}, N={N})")

        i = 0
        beta_r     = train_vec[i:i+3*L].reshape(3, L); i += 3*L
        beta_lam_r = train_vec[i:i+3*L].reshape(3, L); i += 3*L
        beta_lam_v = train_vec[i:i+3*L].reshape(3, L); i += 3*L
        mu_gs      = train_vec[i:i+N].reshape(N);      i += N
        mu_a       = train_vec[i:i+N].reshape(N);      i += N

        return cls(
            betas=CEBetas(beta_r=beta_r, beta_lam_r=beta_lam_r, beta_lam_v=beta_lam_v),
            mu_gs=mu_gs,
            mu_a=mu_a,
        )


# ----------------------------
# small helpers
# ----------------------------

def _solve_ridge_ls(A: torch.Tensor, Y: torch.Tensor, ridge: float) -> torch.Tensor:
    """
    Solve (A^T A + ridge I) X = A^T Y
    A: (N,L)
    Y: (N,K)
    returns X: (L,K)
    """
    N, L = A.shape
    AtA = A.T @ A + float(ridge) * torch.eye(L, device=A.device, dtype=A.dtype)
    AtY = A.T @ Y
    return torch.linalg.solve(AtA, AtY)


def _ridge_ls_sigma_to_target(
    sigma: torch.Tensor,   # (N,L)
    target: torch.Tensor,  # (N,)
    ridge: float,
) -> torch.Tensor:
    N, L = sigma.shape
    AtA = sigma.T @ sigma + float(ridge) * torch.eye(L, device=sigma.device, dtype=sigma.dtype)
    AtY = sigma.T @ target.reshape(N, 1)
    beta = torch.linalg.solve(AtA, AtY)  # (L,1)
    return beta.reshape(-1)              # (L,)


def _apply_lin(M: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    M: (3,3) or (N,3,3)
    x: (N,3)
    return: (N,3)
    """
    if M.ndim == 2:
        return x @ M.T
    if M.ndim == 3:
        return torch.bmm(M, x.unsqueeze(-1)).squeeze(-1)
    raise ValueError(f"M must be (3,3) or (N,3,3), got {tuple(M.shape)}")


def control_from_rva(
    r: torch.Tensor,          # (N,3)
    v: torch.Tensor,          # (N,3)
    a: torch.Tensor,          # (N,3)  a = vdot(t)
    mu_a: torch.Tensor,       # (N,) or scalar tensor/float
    M: torch.Tensor,          # (3,3) or (N,3,3)
    Nmat: torch.Tensor,       # (3,3) or (N,3,3)
    *,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    u      = a - M r - N v
    a_c    = ||u||
    a_c_hat= u / ||u||
    lambda_v = -(a_c + mu_a) * a_c_hat   (only where a_c>eps; else 0)
    """
    if r.ndim != 2 or v.ndim != 2 or a.ndim != 2 or r.shape[-1] != 3 or v.shape[-1] != 3 or a.shape[-1] != 3:
        raise ValueError("r, v, a must be (N,3)")

    Mr = _apply_lin(M, r)
    Nv = _apply_lin(Nmat, v)
    u = a - Mr - Nv

    a_c = torch.linalg.norm(u, dim=-1)  # (N,)
    a_c_hat = u / a_c.clamp_min(eps).unsqueeze(-1)

    if not torch.is_tensor(mu_a):
        mu_a = torch.tensor(float(mu_a), device=r.device, dtype=r.dtype)
    else:
        mu_a = mu_a.to(device=r.device, dtype=r.dtype)

    mu_a = mu_a.expand_as(a_c) if mu_a.ndim == 0 else mu_a
    if mu_a.shape != a_c.shape:
        raise ValueError(f"mu_a must be scalar or shape (N,), got {tuple(mu_a.shape)} vs {tuple(a_c.shape)}")

    norm_lam = a_c + mu_a
    lam_v = torch.zeros_like(u)
    mask = a_c > eps
    lam_v[mask] = -(norm_lam[mask].unsqueeze(-1) * a_c_hat[mask])

    return u, a_c, a_c_hat, lam_v


def lam_r_from_lam_v(
    lam_v: torch.Tensor,          # (N,3)
    lam_v_dot: torch.Tensor,      # (N,3)
    Nmat: torch.Tensor,       # (3,3) or (N,3,3)
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    lam_r = -N^T lam_v - lam_v_dot
    """
    if lam_v.ndim != 2 or lam_v_dot.ndim != 2 or lam_v.shape[-1] != 3 or lam_v_dot.shape[-1] != 3:
        raise ValueError("lam_v and lam_v_dot must be (N,3)")
    N_T_lam_v = _apply_lin(Nmat.transpose(-2,-1), lam_v)
    lam_r = - (N_T_lam_v + lam_v_dot)
    return lam_r


def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    return data


def cone_from_yaml(cfg: Dict[str, Any], *, rf: torch.Tensor, device, dtype) -> ConeConstraint:
    c = cfg["constraints"]["cone"]
    gamma = float(c["gamma_max_deg"])
    n_hat = torch.tensor(c["n_hat"], device=device, dtype=dtype).reshape(3)

    act_cfg = c.get("activation", {})
    # your YAML: activation.tc_s, activation.k
    t_c = float(act_cfg["tc_s"])
    k = float(act_cfg.get("k", 1000.0))
    act = ConeActivation(mode="time", t_c=t_c, k=k)

    eps = float(c.get("eps", 1e-12))
    return ConeConstraint(
        rf=rf.to(device=device, dtype=dtype),
        n_hat=n_hat,
        gamma_max_deg=gamma,
        activation=act,
        eps=eps,
    )


# ----------------------------
# beta_r builders
# ----------------------------

def _beta_r_zero(model: CEModel) -> torch.Tensor:
    L = model.basis.w.numel()
    return torch.zeros((3, L), device=model.device, dtype=model.dtype)


def _beta_r_linear(
    model: CEModel,
    t: torch.Tensor,
    *,
    r0: torch.Tensor, v0: torch.Tensor, rf: torch.Tensor, vf: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """
    Fit beta_r so that r(t) follows a straight line between r0 and rf in time,
    while CE boundary-cubic part r_bc is preserved.
    """
    device = model.device
    dtype = model.dtype

    t = t.to(device=device, dtype=dtype).reshape(-1)
    N = t.numel()
    L = model.basis.w.numel()

    r0 = r0.to(device=device, dtype=dtype).reshape(3)
    v0 = v0.to(device=device, dtype=dtype).reshape(3)
    rf = rf.to(device=device, dtype=dtype).reshape(3)
    vf = vf.to(device=device, dtype=dtype).reshape(3)

    z = model.time_map.t_to_z(t)
    Om = omegas_A6(z, model.time_map.z0, model.time_map.zf)

    sigma = model.basis.sigma(z)  # (N,L)
    sigma0 = model._sigma0.unsqueeze(0)
    sigmaf = model._sigmaf.unsqueeze(0)
    dsigma0 = model._dsigma0.unsqueeze(0)
    dsigmaf = model._dsigmaf.unsqueeze(0)

    b2 = float(model.time_map.b2)

    A = (
        sigma
        - Om.O1.unsqueeze(-1) * sigma0
        - Om.O2.unsqueeze(-1) * sigmaf
        - Om.O3.unsqueeze(-1) * dsigma0
        - Om.O4.unsqueeze(-1) * dsigmaf
    )  # (N,L)

    r_bc = (
        Om.O1.unsqueeze(-1) * r0.unsqueeze(0)
        + Om.O2.unsqueeze(-1) * rf.unsqueeze(0)
        + (Om.O3.unsqueeze(-1) * v0.unsqueeze(0) + Om.O4.unsqueeze(-1) * vf.unsqueeze(0)) / b2
    )  # (N,3)

    alpha = (t - float(model.time_map.t0)) / (float(model.time_map.tf) - float(model.time_map.t0))
    r_lin = r0.unsqueeze(0) + alpha.unsqueeze(-1) * (rf - r0).unsqueeze(0)  # (N,3)

    Y = r_lin - r_bc  # (N,3)
    beta_r_T = _solve_ridge_ls(A, Y, ridge=ridge)  # (L,3)
    beta_r = beta_r_T.T  # (3,L)
    if beta_r.shape != (3, L):
        raise RuntimeError("beta_r shape mismatch")
    return beta_r


# ----------------------------
# main unified init
# ----------------------------

def init_guess(
    model: CEModel,
    t: torch.Tensor,
    *,
    coeffs: SSDynamicsCoeffs,
    r0: torch.Tensor, v0: torch.Tensor, rf: torch.Tensor, vf: torch.Tensor,
    mode: Literal["zero", "linear"] = "linear",
    mu_gs_init: float = 1.0,
    mu_a_init: float = 1e-3,  # each element for (N,) vector
    ridge: float = 1e-8,
    eps: float = 1e-12,
) -> TrainParams:
    """
    Unified init:
      - mode='zero': beta_r=0 (Hermite baseline)
      - mode='linear': beta_r fitted to straight-line r(t)
    Then common pipeline:
      r,v,a from CE -> u=a-Mr-Nv -> a_c,a_hat -> lam_v = -(a_c+mu_a)*a_hat
      fit beta_lam_v via ridge LS on sigma(z(t))
      fit beta_lam_r as constant via ridge LS
      mu_a returned as (N,) vector
    """
    device = model.device
    dtype = model.dtype
    cfg = load_yaml(Path(ROOT / "configs/rendezvous_iv.yaml"))

    t = t.to(device=device, dtype=dtype).reshape(-1)
    N = t.numel()
    L = model.basis.w.numel()

    # --- choose beta_r ---
    if mode == "zero":
        beta_r = _beta_r_zero(model)
    elif mode == "linear":
        beta_r = _beta_r_linear(model, t, r0=r0, v0=v0, rf=rf, vf=vf, ridge=ridge)
    else:
        raise ValueError(f"unknown mode: {mode}")

    # --- evaluate r,v,a (costates don't matter here) ---
    zeros = torch.zeros((3, L), device=device, dtype=dtype)
    tmp_betas = CEBetas(beta_r=beta_r, beta_lam_r=zeros, beta_lam_v=zeros)
    out = model.eval(t, tmp_betas, r0=r0, v0=v0, rf=rf, vf=vf)

    r_guess = out.r
    v_guess = out.v
    a_guess = out.a  # vdot

    # --- build u, a_c, lam_v ---
    M_ = coeffs.M.to(device=device, dtype=dtype)
    N_ = coeffs.N.to(device=device, dtype=dtype)

    mu_a_vec = torch.full((N,), float(mu_a_init), device=device, dtype=dtype)
    u, a_c, a_c_hat, lam_v_guess = control_from_rva(
        r=r_guess, v=v_guess, a=a_guess, mu_a=mu_a_vec, M=M_, Nmat=N_, eps=eps
    )

    # --- sigma(z(t)) ---
    z = model.time_map.t_to_z(t)
    sigma = model.basis.sigma(z)  # (N,L)

    # --- fit beta_lam_v (sigma @ B^T ~ lam_v) ---
    lam_v_T = _solve_ridge_ls(sigma, lam_v_guess, ridge=ridge)  # (L,3)
    beta_lam_v = lam_v_T.T  # (3,L)

    tmp_betas = CEBetas(beta_r=beta_r, beta_lam_r=zeros, beta_lam_v=beta_lam_v)
    out = model.eval(t, tmp_betas, r0=r0, v0=v0, rf=rf, vf=vf)
    lam_v_dot_guess = out.lam_v_dot

    lam_r_guess = lam_r_from_lam_v(
        lam_v=lam_v_guess,
        lam_v_dot=lam_v_dot_guess,
        Nmat=N_,
        eps=eps
    )

    # --- fit beta_lam_r (sigma @ B^T ~ lam_r) ---
    lam_r_T = _solve_ridge_ls(sigma, lam_r_guess, ridge=ridge)  # (L,3)
    beta_lam_r = lam_r_T.T  # (3,L)

    tmp_betas = CEBetas(beta_r=beta_r, beta_lam_r=beta_lam_r, beta_lam_v=beta_lam_v)
    betas = tmp_betas
    mu_gs_vec = torch.full((N,), float(mu_gs_init), device=device, dtype=dtype)

    return TrainParams(betas=betas, mu_a=mu_a_vec, mu_gs=mu_gs_vec)