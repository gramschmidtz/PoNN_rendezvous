# src/lm_solver.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple, List

import torch


@dataclass
class LMSolverConfig:
    damping0: float = 1e-2
    damping_min: float = 1e-20
    damping_max: float = 1e+20
    damp_increase: float = 10.0
    damp_decrease: float = 0.3
    max_iters: int = 200
    grad_inf_tol: float = 1e-10
    step_inf_tol: float = 1e-12
    verbose: bool = True


@dataclass
class LMIter:
    it: int
    cost: float
    g_inf: float
    step_inf: float
    damping: float
    accepted: bool


def _norm_inf(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(x.detach().abs().max().item())


def _ensure_1d(name: str, x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D (shape (k,)), got shape={tuple(x.shape)}")
    return x


def _compute_loss_and_J(
    lossvector: Callable[[torch.Tensor], torch.Tensor],
    trainvector: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute loss vector and Jacobian

    Inputs:
        trainvector: (p,) 1D optimization variable
        lossvector(trainvector): (m,) 1D loss/residual vector

    Returns:
      l: (m,)
      J: (m,p) where J[i,:] = d l_i /d trainvector
    """

    x = _ensure_1d("trainvector",trainvector.detach().clone()).requires_grad_(True)

    with torch.enable_grad():
        l = lossvector(x)
    
    if not torch.is_tensor(l):
        raise TypeError(f"lossvector must return torch.Tensor, got {type(l)}")
    if l.ndim != 1:
        raise ValueError(f"lossvector must return 1D tensor (m,), got shape={tuple(l.shape)}")

    m = l.numel()
    p = x.numel()

    # Edge cases
    if m == 0:
        J = torch.empty((0, p), device=x.device, dtype=x.dtype)
        return l.detach(), J.detach()

    J = torch.empty((m, p), device=x.device, dtype=x.dtype)

    # Compute each row: d r_i / d x (simple/robust; O(m) backward calls)
    for i in range(m):
        (gi,) = torch.autograd.grad(
            l[i],
            x,
            retain_graph=(i != m - 1),
            allow_unused=False,
        )
        gi = _ensure_1d("grad_row", gi)
        if gi.ndim != 1 or gi.numel() != p:
            raise RuntimeError(
                f"Internal error: grad row has shape={tuple(gi.shape)} expected ({p},)"
            )
        J[i] = gi

    return l.detach(), J.detach()


def solve_lm(
    trainvector0: torch.Tensor,
    lossvector: Callable[[torch.Tensor], torch.Tensor],
    cfg: LMSolverConfig,
    callback: Callable[[int, torch.Tensor, torch.Tensor, float, float, float], None] | None = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Levenberg-Marquardt to drive lossvector(trainvector) -> 0.

    Minimizes:
      0.5 * ||lossvector(trainvector)||_2^2

    Contract:
      - trainvector0 must be 1D tensor (p,)
      - lossvector(trainvector) returns a 1D tensor (m,)
    
    Returns:
      best_trainvector, info
    """

    # Enforce 1D optimization variable
    x = _ensure_1d("trainvector0", trainvector0.detach().clone())
    
    lam = float(cfg.damping0)

    # initial
    l, J = _compute_loss_and_J(lossvector, x)
    cost = float(0.5 * (l @ l).item()) if l.numel() > 0 else 0.0
    g = (J.T @ l) if l.numel() > 0 else torch.zeros_like(x)
    g_inf = _norm_inf(g)

    hist: List[LMIter] = []
    if cfg.verbose:
        print(f"[LM] it={0:4d} cost={cost:.6e} |g|inf={g_inf:.6e} lam={lam:.3e}")
    
    if callback is not None:
        callback(0, x, l, cost, g_inf, lam)

    best = {"x": x.clone(), "cost": cost, "it": 0}

    for it in range(1, int(cfg.max_iters) + 1):
        # stop on gradient
        if g_inf <= float(cfg.grad_inf_tol):
            hist.append(LMIter(it=it, cost=cost, g_inf=g_inf, step_inf=0.0, damping=lam, accepted=True))
            if cfg.verbose:
                print(f"[LM] STOP grad tol: it={it} |g|inf={g_inf:.3e}")
            break

        # Normal equations:
        # (J^T J + lam I) dx = - J^T l
        JTJ = J.T @ J
        A = JTJ + lam * torch.eye(JTJ.shape[0], device=JTJ.device, dtype=JTJ.dtype)
        b = -(J.T @ l)

        try:
            dx = torch.linalg.solve(A, b)
        except RuntimeError:
            lam = min(float(cfg.damping_max), lam * float(cfg.damp_increase))
            hist.append(LMIter(it=it, cost=cost, g_inf=g_inf, step_inf=float("inf"), damping=lam, accepted=False))
            if cfg.verbose:
                print(f"[LM] it={it:4d} SOLVE FAIL -> lam={lam:.3e}")
            continue

        dx = _ensure_1d("dx", dx)
        step_inf = _norm_inf(dx)

        # stop on tiny step
        if step_inf <= float(cfg.step_inf_tol):
            hist.append(LMIter(it=it, cost=cost, g_inf=g_inf, step_inf=step_inf, damping=lam, accepted=True))
            if cfg.verbose:
                print(f"[LM] STOP step tol: it={it} |dx|inf={step_inf:.3e}")
            break

        x_trial = x + dx

        l_trial, J_trial = _compute_loss_and_J(lossvector, x_trial)
        cost_trial = float(0.5 * (l_trial @ l_trial).item()) if l_trial.numel() > 0 else 0.0

        accepted = cost_trial < cost

        if accepted:
            x, l, J, cost = x_trial, l_trial, J_trial, cost_trial
            g = (J.T @ l) if l.numel() > 0 else torch.zeros_like(x)
            g_inf = _norm_inf(g)

            lam = max(float(cfg.damping_min), lam * float(cfg.damp_decrease))
            if cost < best["cost"]:
                best = {"x": x.clone(), "cost": cost, "it": it}
        else:
            lam = min(float(cfg.damping_max), lam * float(cfg.damp_increase))

        hist.append(LMIter(it=it, cost=cost, g_inf=g_inf, step_inf=step_inf, damping=lam, accepted=accepted))

        if cfg.verbose:
            tag = "ACC" if accepted else "REJ"
            print(
                f"[LM] it={it:4d} cost={cost:.6e} |g|inf={g_inf:.6e} "
                f"|dx|inf={step_inf:.6e} lam={lam:.3e} {tag}"
            )
        
        if callback is not None:
            callback(it, x, l, cost, g_inf, lam)

    info: Dict[str, Any] = {
        "best_cost": best["cost"],
        "best_it": best["it"],
        "history": hist,
        "final_cost": cost,
        "final_it": (hist[-1].it if hist else 0),
    }
    return best["x"], info