# src/ponn/ce.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Literal

import torch


# ----------------------------
# Activation + derivatives
# ----------------------------

def _act_atan(x: torch.Tensor) -> torch.Tensor:
    return torch.atan(x)

def _d_act_atan(x: torch.Tensor) -> torch.Tensor:
    # d/dx atan(x) = 1/(1+x^2)
    return 1.0 / (1.0 + x * x)

def _dd_act_atan(x: torch.Tensor) -> torch.Tensor:
    # d2/dx2 atan(x) = -2x/(1+x^2)^2
    denom = (1.0 + x * x)
    return (-2.0 * x) / (denom * denom)


ActivationName = Literal["atan"]


# ----------------------------
# Switching functions Ω (Table A6)
# ----------------------------

@dataclass(frozen=True)
class OmegaPack:
    O1: torch.Tensor
    O2: torch.Tensor
    O3: torch.Tensor
    O4: torch.Tensor
    dO1: torch.Tensor
    dO2: torch.Tensor
    dO3: torch.Tensor
    dO4: torch.Tensor
    ddO1: torch.Tensor
    ddO2: torch.Tensor
    ddO3: torch.Tensor
    ddO4: torch.Tensor


def omegas_A6(z: torch.Tensor, z0: float, zf: float) -> OmegaPack:
    """
    Table A6 switching functions for 4 constraints:
      f(z0), f(zf), f'(z0), f'(zf) on z in [z0, zf].

    Uses z* = z - z0, Δz = zf - z0.
    Returns Ω, Ω', Ω'' w.r.t z (NOT time).
    """
    Dz = float(zf - z0)
    if abs(Dz) < 1e-15:
        raise ValueError("zf - z0 must be non-zero")

    zstar = z - float(z0)

    Dz2 = Dz * Dz
    Dz3 = Dz2 * Dz

    # Ω
    O1 = 1.0 + 2.0 * (zstar**3) / Dz3 - 3.0 * (zstar**2) / Dz2
    O2 = -2.0 * (zstar**3) / Dz3 + 3.0 * (zstar**2) / Dz2
    O3 = zstar + (zstar**3) / Dz2 - 2.0 * (zstar**2) / Dz
    O4 = (zstar**3) / Dz2 - (zstar**2) / Dz

    # Ω'
    dO1 = 6.0 * (zstar**2) / Dz3 - 6.0 * zstar / Dz2
    dO2 = -6.0 * (zstar**2) / Dz3 + 6.0 * zstar / Dz2
    dO3 = 1.0 + 3.0 * (zstar**2) / Dz2 - 4.0 * zstar / Dz
    dO4 = 3.0 * (zstar**2) / Dz2 - 2.0 * zstar / Dz

    # Ω''
    ddO1 = 12.0 * zstar / Dz3 - 6.0 / Dz2
    ddO2 = -12.0 * zstar / Dz3 + 6.0 / Dz2
    ddO3 = 6.0 * zstar / Dz2 - 4.0 / Dz
    ddO4 = 6.0 * zstar / Dz2 - 2.0 / Dz

    return OmegaPack(O1, O2, O3, O4, dO1, dO2, dO3, dO4, ddO1, ddO2, ddO3, ddO4)


# ----------------------------
# Mapping t <-> z  (Eq. 10,11)
# ----------------------------

@dataclass(frozen=True)
class TimeMap:
    t0: float
    tf: float
    z0: float
    zf: float

    @property
    def b2(self) -> float:
        # Eq.(11): b^2 = (zf-z0)/(tf-t0) = dz/dt
        dt = self.tf - self.t0
        if abs(dt) < 1e-15:
            raise ValueError("tf - t0 must be non-zero")
        return (self.zf - self.z0) / dt

    def t_to_z(self, t: torch.Tensor) -> torch.Tensor:
        # Eq.(10): z = z0 + (zf-z0)/(tf-t0) * (t-t0)
        return self.z0 + (self.zf - self.z0) * (t - self.t0) / (self.tf - self.t0)

    def z_to_t(self, z: torch.Tensor) -> torch.Tensor:
        # t = t0 + (tf-t0)/(zf-z0) * (z-z0)
        return self.t0 + (self.tf - self.t0) * (z - self.z0) / (self.zf - self.z0)


# ----------------------------
# ELM basis σ(z) and σ', σ''
# ----------------------------

@dataclass
class ELMBasis:
    """
    Single-input ELM basis:
      σ_q(z) = act(w_q z + b_q),  q=1..L
    """
    w: torch.Tensor  # (L,)
    b: torch.Tensor  # (L,)
    activation: ActivationName = "atan"

    def _preact(self, z: torch.Tensor) -> torch.Tensor:
        # z: (...,) -> (...,L)
        return z.unsqueeze(-1) * self.w + self.b

    def sigma(self, z: torch.Tensor) -> torch.Tensor:
        x = self._preact(z)
        if self.activation == "atan":
            return _act_atan(x)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def dsigma_dz(self, z: torch.Tensor) -> torch.Tensor:
        x = self._preact(z)
        if self.activation == "atan":
            # dσ/dz = dσ/dx * dx/dz = act'(x) * w
            return _d_act_atan(x) * self.w
        raise ValueError(f"Unsupported activation: {self.activation}")

    def ddsigma_dz2(self, z: torch.Tensor) -> torch.Tensor:
        x = self._preact(z)
        if self.activation == "atan":
            # d2σ/dz2 = act''(x) * (dx/dz)^2 = act''(x) * w^2
            return _dd_act_atan(x) * (self.w * self.w)
        raise ValueError(f"Unsupported activation: {self.activation}")


# ----------------------------
# CE evaluator (Eq. 53-59)
# ----------------------------

@dataclass
class CEBetas:
    """
    Trainable output weights.
      beta_r:      (3,L) for r_i CE free part
      beta_lam_r:  (3,L) for λ_r,i
      beta_lam_v:  (3,L) for λ_v,i
    """
    beta_r: torch.Tensor
    beta_lam_r: torch.Tensor
    beta_lam_v: torch.Tensor

    def to(self, *, device=None, dtype=None) -> "CEBetas":
        return CEBetas(
            beta_r=self.beta_r.to(device=device, dtype=dtype),
            beta_lam_r=self.beta_lam_r.to(device=device, dtype=dtype),
            beta_lam_v=self.beta_lam_v.to(device=device, dtype=dtype),
        )


@dataclass(frozen=True)
class CEOutput:
    t: torch.Tensor      # (N,)
    z: torch.Tensor      # (N,)
    r: torch.Tensor      # (N,3)
    v: torch.Tensor      # (N,3)
    a: torch.Tensor      # (N,3)  (a = vdot)
    lam_r: torch.Tensor  # (N,3)
    lam_r_dot: torch.Tensor  # (N,3)
    lam_v: torch.Tensor  # (N,3)
    lam_v_dot: torch.Tensor  # (N,3)


class CEModel:
    """
    Implements Eq.(53)-(59) for r,v,a and costates using:
      - ELM basis σ(z) of length L
      - Switching functions Ω1..Ω4 (Table A6)
      - time mapping z(t) (Eq.10-11)

    Important: Ω derivatives are w.r.t z.
    Time derivatives introduce factors b^2, b^4 (Eq.12).
    """

    def __init__(
        self,
        *,
        time_map: TimeMap,
        basis: ELMBasis,
        device=None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.time_map = time_map
        self.basis = ELMBasis(
            w=basis.w.to(device=device, dtype=dtype),
            b=basis.b.to(device=device, dtype=dtype),
            activation=basis.activation,
        )
        self.device = device
        self.dtype = dtype

        # Precompute σ0, σf, σ0', σf' for Eq.(53)-(55)
        z0_t = torch.tensor(self.time_map.z0, device=device, dtype=dtype)
        zf_t = torch.tensor(self.time_map.zf, device=device, dtype=dtype)

        self._sigma0 = self.basis.sigma(z0_t).reshape(-1)          # (L,)
        self._sigmaf = self.basis.sigma(zf_t).reshape(-1)          # (L,)
        self._dsigma0 = self.basis.dsigma_dz(z0_t).reshape(-1)     # (L,)
        self._dsigmaf = self.basis.dsigma_dz(zf_t).reshape(-1)     # (L,)

    @staticmethod
    def from_yaml(cfg: Dict[str, Any], *, seed: int = 0, device=None, dtype=torch.float64) -> "CEModel":
        # time map
        t0 = float(cfg["time"]["t0"])
        tf = float(cfg["time"]["tf"])
        z0 = float(cfg["discretization"]["z0"])
        zf = float(cfg["discretization"]["zf"])
        tm = TimeMap(t0=t0, tf=tf, z0=z0, zf=zf)

        # basis params (ELM random weights)
        L = int(cfg["elm"]["n_hidden"])
        activation = str(cfg["elm"]["activation"]).lower()
        if activation != "atan":
            raise ValueError("This CEModel currently implements activation=atan only")

        ws = cfg["elm"]["weight_sampling"]
        w_min = float(ws["w_min"])
        w_max = float(ws["w_max"])
        c_min = float(ws["c_min"])
        c_max = float(ws["c_max"])

        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))

        w = (w_min + (w_max - w_min) * torch.rand(L, generator=g)).to(device=device, dtype=dtype)
        c = (c_min + (c_max - c_min) * torch.rand(L, generator=g)).to(device=device, dtype=dtype)
        b = - w * c

        basis = ELMBasis(w=w, b=b, activation="atan")
        return CEModel(time_map=tm, basis=basis, device=device, dtype=dtype)

    def eval(
        self,
        t: torch.Tensor,          # (N,) or (...,)
        betas: CEBetas,
        *,
        r0: torch.Tensor, v0: torch.Tensor, rf: torch.Tensor, vf: torch.Tensor,  # each (3,)
    ) -> CEOutput:
        """
        Evaluate CE states/costates on time grid t using Eq.(53)-(59).
        """
        # ensure shapes
        if r0.shape != (3,) or v0.shape != (3,) or rf.shape != (3,) or vf.shape != (3,):
            raise ValueError("r0,v0,rf,vf must be shape (3,)")

        t = t.to(device=self.device, dtype=self.dtype).reshape(-1)  # (N,)
        betas = betas.to(device=self.device, dtype=self.dtype)

        N = t.numel()
        L = self.basis.w.numel()

        if betas.beta_r.shape != (3, L):
            raise ValueError(f"beta_r must be (3,{L})")
        if betas.beta_lam_r.shape != (3, L):
            raise ValueError(f"beta_lam_r must be (3,{L})")
        if betas.beta_lam_v.shape != (3, L):
            raise ValueError(f"beta_lam_v must be (3,{L})")

        # map to z
        z = self.time_map.t_to_z(t)  # (N,)

        # Ω(z)
        Om = omegas_A6(z, self.time_map.z0, self.time_map.zf)

        # σ(z), σ'(z), σ''(z) wrt z
        sigma = self.basis.sigma(z)            # (N,L)
        dsigma = self.basis.dsigma_dz(z)       # (N,L)
        ddsigma = self.basis.ddsigma_dz2(z)    # (N,L)

        # constants
        b2 = float(self.time_map.b2)           # dz/dt
        b4 = b2 * b2

        # expand σ0 etc for broadcasting
        sigma0 = self._sigma0.unsqueeze(0)     # (1,L)
        sigmaf = self._sigmaf.unsqueeze(0)     # (1,L)
        dsigma0 = self._dsigma0.unsqueeze(0)   # (1,L)
        dsigmaf = self._dsigmaf.unsqueeze(0)   # (1,L)

        # ---- Eq.(53): r ----
        # free part vector: (N,L)
        free_r = sigma \
                 - Om.O1.unsqueeze(-1) * sigma0 \
                 - Om.O2.unsqueeze(-1) * sigmaf \
                 - Om.O3.unsqueeze(-1) * dsigma0 \
                 - Om.O4.unsqueeze(-1) * dsigmaf

        # (N,3) = free_r (N,L) @ beta_r^T (L,3)
        r_free = free_r @ betas.beta_r.T

        r_bc = (
            Om.O1.unsqueeze(-1) * r0.unsqueeze(0)
            + Om.O2.unsqueeze(-1) * rf.unsqueeze(0)
            + (Om.O3.unsqueeze(-1) * v0.unsqueeze(0) + Om.O4.unsqueeze(-1) * vf.unsqueeze(0)) / b2
        )
        r = r_free + r_bc

        # ---- Eq.(54): v ----
        free_v = dsigma \
                 - Om.dO1.unsqueeze(-1) * sigma0 \
                 - Om.dO2.unsqueeze(-1) * sigmaf \
                 - Om.dO3.unsqueeze(-1) * dsigma0 \
                 - Om.dO4.unsqueeze(-1) * dsigmaf
        v_free = free_v @ betas.beta_r.T

        v_bc = (
            Om.dO1.unsqueeze(-1) * r0.unsqueeze(0)
            + Om.dO2.unsqueeze(-1) * rf.unsqueeze(0)
            + (Om.dO3.unsqueeze(-1) * v0.unsqueeze(0) + Om.dO4.unsqueeze(-1) * vf.unsqueeze(0)) / b2
        )
        v = (b2 * (v_free + v_bc))  # because dr/dt = (dz/dt) dr/dz = b2 * ...

        # ---- Eq.(55): a ----
        free_a = ddsigma \
                 - Om.ddO1.unsqueeze(-1) * sigma0 \
                 - Om.ddO2.unsqueeze(-1) * sigmaf \
                 - Om.ddO3.unsqueeze(-1) * dsigma0 \
                 - Om.ddO4.unsqueeze(-1) * dsigmaf
        a_free = free_a @ betas.beta_r.T

        a_bc = (
            Om.ddO1.unsqueeze(-1) * r0.unsqueeze(0)
            + Om.ddO2.unsqueeze(-1) * rf.unsqueeze(0)
            + (Om.ddO3.unsqueeze(-1) * v0.unsqueeze(0) + Om.ddO4.unsqueeze(-1) * vf.unsqueeze(0)) / b2
        )
        a = (b4 * (a_free + a_bc))  # d2r/dt2 = (dz/dt)^2 d2r/dz2 = b2^2 * ...

        # ---- Eq.(56)-(59): costates and time-derivatives ----
        # λ_r = σ^T β_{λr}
        lam_r = sigma @ betas.beta_lam_r.T             # (N,3)
        lam_v = sigma @ betas.beta_lam_v.T             # (N,3)

        # \dot λ = b^2 σ'^T β  (Eq.57,59)  (because d/dt = b2 d/dz)
        lam_r_dot = (b2 * (dsigma @ betas.beta_lam_r.T))
        lam_v_dot = (b2 * (dsigma @ betas.beta_lam_v.T))

        return CEOutput(
            t=t,
            z=z,
            r=r,
            v=v,
            a=a,
            lam_r=lam_r,
            lam_r_dot=lam_r_dot,
            lam_v=lam_v,
            lam_v_dot=lam_v_dot,
        )