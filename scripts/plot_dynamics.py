# scripts/plot_dynamics.py
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

from src.ponn.dynamics import SSOrbitParams, compute_ss_coeffs
from src.ponn.propagate import propagate_ss_rv


# -------------------------
# Config helpers
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
    r0 = np.array(b["r0_km"], dtype=float)
    v0 = np.array(b["v0_km_s"], dtype=float)
    rf = np.array(b["rf_km"], dtype=float)
    vf = np.array(b["vf_km_s"], dtype=float)
    if r0.shape != (3,) or v0.shape != (3,) or rf.shape != (3,) or vf.shape != (3,):
        raise ValueError("boundary vectors must be length-3")
    return r0, v0, rf, vf


def timegrid_from_yaml(cfg: Dict[str, Any], n: int | None = None) -> np.ndarray:
    t0 = float(cfg["time"]["t0"])
    tf = float(cfg["time"]["tf"])
    if n is None:
        # YAML has n_dis (collocation points). For propagation, a bit denser is nicer.
        n_dis = int(cfg["discretization"]["n_dis"])
        n = max(3, 10 * n_dis + 1)  # e.g. 501 for n_dis=50
    return np.linspace(t0, tf, n, dtype=float)


# -------------------------
# Controls
# -------------------------

def make_u_grid(
    t: np.ndarray,
    mode: str,
    *,
    u_max: float,
    axis: str = "y",
    amp: float = 1.0,
    freq_hz: float = 1 / 60.0,
    phase: float = 0.0,
    pulse_t0: float = 50.0,
    pulse_t1: float = 100.0,
) -> np.ndarray:
    """
    Returns u_grid with shape (N,3), in km/s^2.
    amp is a multiplier on u_max (so amp=1 -> |u|<=u_max).
    axis chooses direction: x|y|z or "-x" etc.
    """
    N = t.shape[0]
    u = np.zeros((N, 3), dtype=float)

    # direction
    ax = axis.strip().lower()
    sign = 1.0
    if ax.startswith("-"):
        sign = -1.0
        ax = ax[1:]
    idx = {"x": 0, "y": 1, "z": 2}.get(ax, None)
    if idx is None:
        raise ValueError("axis must be one of x,y,z,-x,-y,-z")

    umax_eff = float(u_max) * float(amp)

    if mode == "zero":
        pass

    elif mode == "const":
        u[:, idx] = sign * umax_eff

    elif mode == "sine":
        # u(t) = umax_eff * sin(2π f t + phase)
        u[:, idx] = sign * umax_eff * np.sin(2.0 * math.pi * float(freq_hz) * t + float(phase))

    elif mode == "pulse":
        # u(t) = umax_eff on [pulse_t0, pulse_t1], else 0
        mask = (t >= float(pulse_t0)) & (t <= float(pulse_t1))
        u[mask, idx] = sign * umax_eff

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Optional: clamp magnitude to u_max (in case of weird combos)
    mag = np.linalg.norm(u, axis=1, keepdims=True)
    too_big = mag > float(u_max)
    if np.any(too_big):
        u[too_big[:, 0]] *= (float(u_max) / (mag[too_big] + 1e-15))

    return u


# -------------------------
# Cone geometry
# -------------------------

def orthonormal_basis_from_axis(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        raise ValueError("axis vector is near zero")
    a = a / n

    # choose a vector not parallel to a
    if abs(a[0]) < 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    else:
        tmp = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(a, tmp)
    e1 /= (np.linalg.norm(e1) + 1e-15)
    e2 = np.cross(a, e1)
    e2 /= (np.linalg.norm(e2) + 1e-15)
    return a, e1, e2


def make_cone_surface(
    apex: np.ndarray,
    axis: np.ndarray,
    gamma_deg: float,
    height: float = 0.2,
    n_s: int = 40,
    n_th: int = 80,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cone with apex at `apex`, axis direction = axis (unit),
    half-angle = gamma_deg, height along axis = height (km).
    Returns X,Y,Z surface arrays for plot_surface.
    """
    apex = np.asarray(apex, dtype=float).reshape(3)
    a, e1, e2 = orthonormal_basis_from_axis(axis)
    gamma = math.radians(float(gamma_deg))
    s_vals = np.linspace(0.0, float(height), n_s)
    th_vals = np.linspace(0.0, 2.0 * math.pi, n_th)

    X = np.zeros((n_s, n_th))
    Y = np.zeros((n_s, n_th))
    Z = np.zeros((n_s, n_th))

    for i, s in enumerate(s_vals):
        radius = s * math.tan(gamma)
        for j, th in enumerate(th_vals):
            p_local = s * a + radius * (math.cos(th) * e1 + math.sin(th) * e2)
            p = apex + p_local
            X[i, j], Y[i, j], Z[i, j] = p[0], p[1], p[2]

    return X, Y, Z


# -------------------------
# Plot
# -------------------------

def set_axes_equal(ax):
    """Make 3D axes have equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_mid = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/rendezvous_iv.yaml")
    ap.add_argument("--n", type=int, default=None, help="number of time samples for propagation")
    ap.add_argument("--mode", type=str, default="zero",
                    choices=["zero", "const", "sine", "pulse"])
    ap.add_argument("--axis", type=str, default="y", help="x,y,z,-x,-y,-z")
    ap.add_argument("--amp", type=float, default=1.0, help="multiplier on control_max")
    ap.add_argument("--freq_hz", type=float, default=1/60.0, help="for sine")
    ap.add_argument("--phase", type=float, default=0.0, help="radians for sine")
    ap.add_argument("--pulse_t0", type=float, default=50.0)
    ap.add_argument("--pulse_t1", type=float, default=100.0)
    ap.add_argument("--cone_height", type=float, default=0.2, help="km")
    ap.add_argument("--show_cone_always", action="store_true",
                    help="if set, show cone regardless of activation tc_s")
    ap.add_argument("--save", type=str, default=None, help="save figure path (png)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg_path = (root / args.config).resolve()
    cfg = load_yaml(cfg_path)

    # orbit + coeffs
    orbit = orbit_from_yaml(cfg)
    coeffs = compute_ss_coeffs(orbit, device="cpu", dtype=torch.float64)

    # boundary
    r0_np, v0_np, rf_np, vf_np = boundary_from_yaml(cfg)

    # time grid
    t_np = timegrid_from_yaml(cfg, n=args.n)

    # control grid (km/s^2)
    u_max = float(cfg["constraints"]["control_max_km_s2"])
    u_np = make_u_grid(
        t_np,
        args.mode,
        u_max=u_max,
        axis=args.axis,
        amp=args.amp,
        freq_hz=args.freq_hz,
        phase=args.phase,
        pulse_t0=args.pulse_t0,
        pulse_t1=args.pulse_t1,
    )

    # propagate using existing module
    t = torch.tensor(t_np, dtype=torch.float64)
    r0 = torch.tensor(r0_np, dtype=torch.float64)
    v0 = torch.tensor(v0_np, dtype=torch.float64)
    u = torch.tensor(u_np, dtype=torch.float64)
    result = propagate_ss_rv(t, r0, v0, u, coeffs)

    # ---- derive acceleration a(t) = vdot from dynamics (using the integrated r,v and the applied u) ----
    # result.r, result.v: torch (N,3), u: torch (N,3)
    with torch.no_grad():
        # vdot = M r + N v + u
        vdot = (result.r @ coeffs.M.T) + (result.v @ coeffs.N.T) + u  # (N,3)
    
    a_np = vdot.detach().cpu().numpy()
    r = result.r.detach().cpu().numpy()

    # cone params
    cone = cfg["constraints"]["cone"]
    gamma_deg = float(cone["gamma_max_deg"])
    n_hat = np.array(cone["n_hat"], dtype=float)
    tc_s = float(cone["activation"]["tc_s"])
    show_cone = args.show_cone_always or (t_np[-1] >= tc_s)

    # plot
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # trajectory
    ax.plot(r[:, 0], r[:, 1], r[:, 2], linewidth=2.0, label="trajectory r(t)")

    # start / target
    ax.scatter([r0_np[0]], [r0_np[1]], [r0_np[2]], s=60, marker="o", label="start r0")
    ax.scatter([rf_np[0]], [rf_np[1]], [rf_np[2]], s=80, marker="*", label="target rf")

    # cone surface
    if show_cone:
        X, Y, Z = make_cone_surface(
        apex=rf_np,
        axis=n_hat,
        gamma_deg=gamma_deg,
        height=float(args.cone_height),
        )
        ax.plot_surface(X, Y, Z, alpha=0.15, linewidth=0.0)

        # cone axis line (from apex)
        a_unit = n_hat / (np.linalg.norm(n_hat) + 1e-15)
        a_end = rf_np + a_unit * float(args.cone_height)
        ax.plot([rf_np[0], a_end[0]], [rf_np[1], a_end[1]], [rf_np[2], a_end[2]],
            linewidth=1.5, label="cone axis")


    # labels
    ax.set_title(f"SS Dynamics Propagation (mode={args.mode}, axis={args.axis}, amp={args.amp})")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    ax.legend(loc="best")

    # nice scaling
    set_axes_equal(ax)

    # also show a 2D time plot of control magnitude (optional but handy)
    # (kept minimal; comment out if you don't want)
    fig2 = plt.figure(figsize=(9, 3))
    ax2 = fig2.add_subplot(111)
    u_mag = np.linalg.norm(u_np, axis=1)
    ax2.plot(t_np, u_mag, linewidth=2.0)
    ax2.set_title("Control magnitude ||u(t)|| [km/s^2]")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("||u||")
    ax2.grid(True, alpha=0.3)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=200, bbox_inches="tight")
        # save control plot too
        fig2.savefig(str(out.with_name(out.stem + "_u.png")), dpi=200, bbox_inches="tight")
    
    # -------------------------
    # 3x4 subplot:
    # rows = x,y,z
    # cols = r, v, a (vdot), a_c (control input)
    # -------------------------
    fig3, axes = plt.subplots(3, 4, figsize=(15, 8), sharex=True)

    comps = ["x", "y", "z"]
    cols = ["r [km]", "v [km/s]", "a = v̇ [km/s²]", "a_c (control) [km/s²]"]

    r_np = result.r.detach().cpu().numpy()
    v_np = result.v.detach().cpu().numpy()
    a_np = a_np           # already computed earlier
    u_np = u_np           # control grid already numpy

    for i in range(3):  # row index (x,y,z)

        # ---- Column 0: r ----
        ax = axes[i, 0]
        ax.plot(t_np, r_np[:, i], linewidth=2.0)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(cols[0])
        ax.set_ylabel(comps[i])

        ax.scatter([t_np[0]], [r0_np[i]], marker="o", s=30)
        ax.scatter([t_np[-1]], [rf_np[i]], marker="*", s=60)

        # ---- Column 1: v ----
        ax = axes[i, 1]
        ax.plot(t_np, v_np[:, i], linewidth=2.0)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(cols[1])

        ax.scatter([t_np[0]], [v0_np[i]], marker="o", s=30)
        ax.scatter([t_np[-1]], [vf_np[i]], marker="*", s=60)

        # ---- Column 2: a = vdot ----
        ax = axes[i, 2]
        ax.plot(t_np, a_np[:, i], linewidth=2.0)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(cols[2])

        # ---- Column 3: a_c = control input ----
        ax = axes[i, 3]
        ax.plot(t_np, u_np[:, i], linewidth=2.0)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(cols[3])

    # bottom row x-labels
    for j in range(4):
        axes[2, j].set_xlabel("t [s]")

    fig3.suptitle(
        "Integrated SS dynamics (rows: x,y,z | cols: r, v, a=v̇, a_c)",
        y=0.98
    )
    fig3.tight_layout()


    plt.show()


if __name__ == "__main__":
    main()