# src/ponn/problem.py
from __future__ import annotations

from dataclasses import dataclass

import torch

from .ce import CEModel
from .losses import RendezvousLosses
from .init_guess import TrainParams

@dataclass
class RendezvousProblem:
    model: CEModel
    t: torch.Tensor
    r0: torch.Tensor
    v0: torch.Tensor
    rf: torch.Tensor
    vf: torch.Tensor
    loss: RendezvousLosses
    L: int
    N: int
    device: torch.device
    dtype: torch.dtype

    def __call__(self, trainvector: torch.Tensor) -> torch.Tensor:
        tp = TrainParams.unpack(
            trainvector,
            self.L,
            self.N
        )

        out = self.model.eval(
            self.t,
            tp.betas,
            r0=self.r0,
            v0=self.v0,
            rf=self.rf,
            vf=self.vf
        )

        blocks = self.loss.compute(
            r=out.r,
            v=out.v,
            vdot=out.a,
            lam_r=out.lam_r,
            lam_r_dot=out.lam_r_dot,
            lam_v=out.lam_v,
            lam_v_dot=out.lam_v_dot,
            mu_gs=tp.mu_gs,
            mu_a=tp.mu_a
        )

        return self.loss.pack(blocks)