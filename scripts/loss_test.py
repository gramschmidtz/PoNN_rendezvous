# scripts/loss_test.py
# Run:
#   python scripts/loss_test.py --config configs/rendezvous_iv.yaml --init linear
#
# This script:
#  1) Loads YAML
#  2) Builds SS coeffs + CEModel
#  3) Builds InitGuess on TRAIN grid (N = n_dis)
#  4) Evaluates CE on TRAIN grid
#  5) Builds RendezvousLosses(t0, tf stored)
#  6) Computes residual blocks + packed residual vector
#  7) Prints shapes + (max, rms) diagnostics

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ponn.ce import CEModel
from src.ponn.dynamics import SSOrbitParams, compute_ss_coeffs
from src.ponn.init_guess import init_guess
from src.ponn.losses import RendezvousLosses, RendezvousLossWeights
from src.ponn.constraint import ConeConstraint, ConeActivation, ControlConstraint


# -------------------------
# YAML helpers
# -------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    return data


def orbit_from_yaml(cfg: Dict[str, Any]) -> SSOrbitParams:
    o = cfg["orbit"]
    it_rad = float(o["it_deg"]) * math.pi / 180.0
    return SSOrbitParams(
        Re_km=float(o["Re_km"]),
        Rt_km=float(o["Rt_km"]),
        it_rad=it_rad,
        mu_km3_s2=float(o["mu_km3_s2"]),
        J2=float(o["J2"]),
    )


def boundary_from_yaml(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b = cfg["boundary"]
    return (
        np.array(b["r0_km"], dtype=float),
        np.array(b["v0_km_s"], dtype=float),
        np.array(b["rf_km"], dtype=float),
        np.array(b["vf_km_s"], dtype=float),
    )


def train_timegrid_from_yaml(cfg: Dict[str, Any]) -> np.ndarray:
    t0 = float(cfg["time"]["t0"])
    tf = float(cfg["time"]["tf"])
    n_dis = int(cfg["discretization"]["n_dis"])  # TRAIN grid length
    return np.linspace(t0, tf, n_dis)


def ctrl_from_yaml(cfg: Dict[str, Any]) -> ControlConstraint:
    c = cfg["constraints"]
    # your YAML: constraints.control_max_km_s2
    a_c_max = float(c["control_max_km_s2"])
    eps = float(c.get("control_eps", 1e-12))
    return ControlConstraint(a_c_max=a_c_max, eps=eps)


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


def weights_from_yaml(cfg: Dict[str, Any]) -> RendezvousLossWeights:
    # optional; if not present, defaults=1.0
    w = cfg.get("loss_weights", None)
    if not isinstance(w, dict):
        return RendezvousLossWeights()
    return RendezvousLossWeights(
        w_v=float(w.get("w_v", 3000.0)),
        w_lam_r=float(w.get("w_lam_r", 1.0)),
        w_lam_v=float(w.get("w_lam_v", 1.0)),
        w_gs=float(w.get("w_gs", 700.0)),
        w_a=float(w.get("w_a", 700.0)),
    )


def stats(name: str, x: torch.Tensor) -> str:
    x = x.detach()
    max_abs = x.abs().max().item()
    cost = 0.5 * (x * x).sum().item()
    return f"{name:7s} shape={tuple(x.shape)!s:12s}  max={max_abs:.3e}  cost={cost:.3e}"


# -------------------------
# main
# -------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/rendezvous_iv.yaml")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--init", type=str, default="linear", choices=["zero", "linear"])
    ap.add_argument("--mu_a", type=float, default=1e-3)
    ap.add_argument("--mu_gs", type=float, default=1.0)
    ap.add_argument("--ridge", type=float, default=1e-8)

    ap.add_argument("--near_tol_km", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    cfg = load_yaml(Path(ROOT / args.config))

    # times
    t0 = float(cfg["time"]["t0"])
    tf = float(cfg["time"]["tf"])
    t_np = train_timegrid_from_yaml(cfg)
    t = torch.tensor(t_np, device=device, dtype=dtype)
    N = t.numel()

    # orbit + coeffs
    orbit = orbit_from_yaml(cfg)
    coeffs = compute_ss_coeffs(orbit, device=device, dtype=dtype)

    # boundary
    r0_np, v0_np, rf_np, vf_np = boundary_from_yaml(cfg)
    r0 = torch.tensor(r0_np, device=device, dtype=dtype)
    v0 = torch.tensor(v0_np, device=device, dtype=dtype)
    rf = torch.tensor(rf_np, device=device, dtype=dtype)
    vf = torch.tensor(vf_np, device=device, dtype=dtype)

    # CE model
    model = CEModel.from_yaml(cfg, seed=args.seed, device=device, dtype=dtype)

    # init guess on TRAIN grid
    guess = init_guess(
        model,
        t,
        coeffs=coeffs,
        r0=r0, v0=v0, rf=rf, vf=vf,
        mode=args.init,
        mu_gs_init=args.mu_gs,
        mu_a_init=args.mu_a,
        ridge=args.ridge
    )

    # eval CE on TRAIN grid
    out = model.eval(t, guess.betas, r0=r0, v0=v0, rf=rf, vf=vf)

    # constraints + weights
    cone = cone_from_yaml(cfg, rf=rf, device=device, dtype=dtype)
    ctrl_cstr = ctrl_from_yaml(cfg)
    weights = weights_from_yaml(cfg)

    losses = RendezvousLosses(
        coeffs=coeffs,
        weights=weights,
        cone_cstr=cone,
        ctrl_cstr=ctrl_cstr,
        t0=t0,
        tf=tf,
        near_tol_km=float(args.near_tol_km),
    )

    # compute residuals (vdot == out.a)
    blocks = losses.compute(
        r=out.r,
        v=out.v,
        vdot=out.a,
        lam_r=out.lam_r,
        lam_r_dot=out.lam_r_dot,
        lam_v=out.lam_v,
        lam_v_dot=out.lam_v_dot,
        mu_a=guess.mu_a,
        mu_gs=guess.mu_gs,
    )
    vec = losses.pack(blocks)

    print(f"[grid] N_train={N}  t0={t0}  tf={tf}")
    print(f"[betas] beta_r={tuple(guess.betas.beta_r.shape)}  beta_lam_r={tuple(guess.betas.beta_lam_r.shape)}  beta_lam_v={tuple(guess.betas.beta_lam_v.shape)}")
    print(f"[mu]   mu_a={tuple(guess.mu_a.shape)}  mu_gs={tuple(guess.mu_gs.shape)}")
    print("")
    print(stats("Lv", blocks["Lv"]))
    print(stats("Llam_r", blocks["Llam_r"]))
    print(stats("Llam_v", blocks["Llam_v"]))
    print(stats("Lgs", blocks["Lgs"]))
    print(stats("La", blocks["La"]))
    print("")

    print("Cost of Lv_x:", (blocks["Lv"][:,0]**2*0.5).sum())
    print("Cost of Lv_y:", (blocks["Lv"][:,1]**2*0.5).sum())
    print("Cost of Lv_z:", (blocks["Lv"][:,2]**2*0.5).sum())
    print("")

    print("Cost of Llamr_x:", (blocks["Llam_r"][:,0]**2*0.5).sum())
    print("Cost of Llamr_y:", (blocks["Llam_r"][:,1]**2*0.5).sum())
    print("Cost of Llamr_z:", (blocks["Llam_r"][:,2]**2*0.5).sum())
    print("")

    print("Cost of Llamv_x:", (blocks["Llam_v"][:,0]**2*0.5).sum())
    print("Cost of Llamv_y:", (blocks["Llam_v"][:,1]**2*0.5).sum())
    print("Cost of Llamv_z:", (blocks["Llam_v"][:,2]**2*0.5).sum())
    print("")

    print("Cost of Lgs:", (blocks["Lgs"]**2*0.5).sum())
    print("Cost of La:", (blocks["La"]**2*0.5).sum())
    print("")

    print(stats("packed", vec))
    print(f"[packed] expected length = 11*N = {11*N}, got {vec.numel()}")

    # sanity asserts
    assert blocks["Lv"].shape == (N, 3)
    assert blocks["Llam_r"].shape == (N, 3)
    assert blocks["Llam_v"].shape == (N, 3)
    assert blocks["Lgs"].shape == (N,)
    assert blocks["La"].shape == (N,)
    assert vec.numel() == 11 * N


if __name__ == "__main__":
    main()