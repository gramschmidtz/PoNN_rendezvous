# scripts/plot_traj_from_checkpoint.py
from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ponn.ce import CEModel
from src.ponn.dynamics import SSOrbitParams, compute_ss_coeffs
from src.ponn.propagate import propagate_ss_rv, lerp_u, hold_u
from src.ponn.ponn import control_from_costate


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


def boundary_from_yaml(cfg: Dict[str, Any]):
    b = cfg["boundary"]
    return (
        np.array(b["r0_km"], dtype=float),
        np.array(b["v0_km_s"], dtype=float),
        np.array(b["rf_km"], dtype=float),
        np.array(b["vf_km_s"], dtype=float),
    )


# -------------------------
# plotting helpers (cone)
# -------------------------
def set_axes_equal(ax):
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


def orthonormal_basis_from_axis(axis: np.ndarray):
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        raise ValueError("axis vector is near zero")
    a = a / n

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
):
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
# run / checkpoint helpers (new structure)
# -------------------------
def is_run_dir(p: Path) -> bool:
    return (p / "tb").is_dir()


def list_run_dirs(exp_dir: Path) -> List[Path]:
    if not exp_dir.exists():
        return []
    runs = [d for d in exp_dir.iterdir() if d.is_dir() and is_run_dir(d)]
    return sorted(runs, key=lambda d: (d.stat().st_mtime, d.name))


def pick_run_dir(exp_dir: Path, run_id: Optional[str]) -> Path:
    if run_id:
        rd = exp_dir / run_id
        if not is_run_dir(rd):
            raise FileNotFoundError(f"run_dir not found or missing tb/: {rd}")
        return rd

    runs = list_run_dirs(exp_dir)
    if not runs:
        raise FileNotFoundError(f"No run dirs found under: {exp_dir}")
    return runs[-1]  # latest


def pick_latest_ckpt(run_dir: Path) -> Path:
    # Prefer checkpoint_*.pt (new naming), fallback to checkpoint.pt
    cand = sorted(run_dir.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime)
    if cand:
        return cand[-1]
    p = run_dir / "checkpoint.pt"
    if p.exists():
        return p
    raise FileNotFoundError(f"No checkpoint_*.pt or checkpoint.pt found in: {run_dir}")


# -------------------------
# checkpoint helper
# -------------------------
def load_checkpoint(path: Path, *, device: torch.device, dtype: torch.dtype):
    ckpt = torch.load(path, map_location="cpu")
    betas = ckpt["betas"]
    beta_r = betas["beta_r"].to(device=device, dtype=dtype)
    beta_lam_r = betas["beta_lam_r"].to(device=device, dtype=dtype)
    beta_lam_v = betas["beta_lam_v"].to(device=device, dtype=dtype)

    mu_gs = ckpt["mu_gs"].to(device=device, dtype=dtype)
    mu_a = ckpt["mu_a"].to(device=device, dtype=dtype)

    return beta_r, beta_lam_r, beta_lam_v, mu_gs, mu_a, ckpt


def save_fig(fig, outpath: Path, *, dpi: int = 250):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outpath}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/rendezvous_iv.yaml")

    # New run selection
    ap.add_argument("--logdir", type=str, default=None, help="Experiment dir: runs/<exp_name>")
    ap.add_argument("--run-dir", type=str, default=None, help="Run dir: runs/<exp_name>/<run_id>")
    ap.add_argument("--run-id", type=str, default=None, help="Pick run_id under --logdir (default: latest)")

    # behavior
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--dtype", type=str, default="float64")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cone_height", type=float, default=0.2)
    ap.add_argument("--show_cone_always", action="store_true")
    ap.add_argument("--u_mode", type=str, default="lerp", choices=["lerp", "hold"])

    # default: save figures; optional show
    ap.add_argument("--show", action="store_true", help="Also display figures interactively")

    args = ap.parse_args()

    if (args.logdir is None) == (args.run_dir is None):
        raise SystemExit("Provide exactly one of --logdir or --run-dir")

    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    cfg = load_yaml(Path(ROOT / args.config))

    # pick run_dir
    if args.run_dir is not None:
        run_dir = Path(args.run_dir).resolve()
        if not is_run_dir(run_dir):
            raise SystemExit(f"--run-dir must contain tb/: {run_dir}")
    else:
        exp_dir = Path(args.logdir).resolve()
        run_dir = pick_run_dir(exp_dir, args.run_id)

    # pick checkpoint (latest)
    ckpt_path = pick_latest_ckpt(run_dir)
    print(f"[run_dir] {run_dir}")
    print(f"[ckpt]    {ckpt_path}")

    outdir = run_dir / "export_plots"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[outdir]  {outdir}")

    # orbit
    orbit = orbit_from_yaml(cfg)
    coeffs = compute_ss_coeffs(orbit, device=device, dtype=dtype)

    # boundary
    r0_np, v0_np, rf_np, vf_np = boundary_from_yaml(cfg)

    t0 = float(cfg["time"]["t0"])
    tf = float(cfg["time"]["tf"])
    n_dis = int(cfg["discretization"]["n_dis"])

    # grids
    t_train_np = np.linspace(t0, tf, n_dis)
    t_prop_np = t_train_np  # keep same grid to isolate u interpolation effects

    t_train = torch.tensor(t_train_np, device=device, dtype=dtype)
    t_prop = torch.tensor(t_prop_np, device=device, dtype=dtype)

    r0 = torch.tensor(r0_np, device=device, dtype=dtype)
    v0 = torch.tensor(v0_np, device=device, dtype=dtype)
    rf = torch.tensor(rf_np, device=device, dtype=dtype)
    vf = torch.tensor(vf_np, device=device, dtype=dtype)

    # CE model
    model = CEModel.from_yaml(cfg, seed=args.seed, device=device, dtype=dtype)

    # load checkpoint
    beta_r, beta_lam_r, beta_lam_v, mu_gs, mu_a, ckpt = load_checkpoint(
        ckpt_path, device=device, dtype=dtype
    )

    # build betas object
    from src.ponn.ce import CEBetas
    betas = CEBetas(beta_r=beta_r, beta_lam_r=beta_lam_r, beta_lam_v=beta_lam_v)

    # eval CE on train grid
    out_tr = model.eval(t_train, betas, r0=r0, v0=v0, rf=rf, vf=vf)

    # control from costate
    ctrl_tr = control_from_costate(out_tr.lam_v, mu_a)
    u_tr = ctrl_tr.u
    a_c_tr = ctrl_tr.a_c

    # build u on prop grid
    if args.u_mode == "lerp":
        u_prop = lerp_u(t_prop, t_train, u_tr)
    elif args.u_mode == "hold":
        u_prop = hold_u(t_prop, t_train, u_tr)
    else:
        raise ValueError(f"unknown u_mode: {args.u_mode}")

    # propagate
    prop = propagate_ss_rv(t_prop, r0, v0, u_prop, coeffs)

    with torch.no_grad():
        a_prop = (prop.r @ coeffs.M.T) + (prop.v @ coeffs.N.T) + u_prop
        a_c_prop = torch.linalg.norm(u_prop, dim=-1)

    # numpy
    r_ce = out_tr.r.detach().cpu().numpy()
    v_ce = out_tr.v.detach().cpu().numpy()
    a_ce = out_tr.a.detach().cpu().numpy()

    lam_r_ce = out_tr.lam_r.detach().cpu().numpy()
    lam_r_dot_ce = out_tr.lam_r_dot.detach().cpu().numpy()
    lam_v_ce = out_tr.lam_v.detach().cpu().numpy()
    lam_v_dot_ce = out_tr.lam_v_dot.detach().cpu().numpy()

    mu_gs_ce = mu_gs.detach().cpu().numpy()
    mu_a_ce = mu_a.detach().cpu().numpy()

    u_ce_np = u_tr.detach().cpu().numpy()
    ac_ce = a_c_tr.detach().cpu().numpy()

    r_pr = prop.r.detach().cpu().numpy()
    v_pr = prop.v.detach().cpu().numpy()
    a_pr = a_prop.detach().cpu().numpy()
    u_pr_np = u_prop.detach().cpu().numpy()
    ac_pr = a_c_prop.detach().cpu().numpy()

    # basename to embed run_id / u_mode
    run_id = run_dir.name
    suffix = f"{run_id}_u-{args.u_mode}"

    # =========================
    # fig1: 3D + cone
    # =========================
    fig1 = plt.figure(figsize=(9, 7))
    ax = fig1.add_subplot(111, projection="3d")

    ax.plot(r_ce[:, 0], r_ce[:, 1], r_ce[:, 2], label="CE (trained)", linewidth=2)
    ax.plot(r_pr[:, 0], r_pr[:, 1], r_pr[:, 2], "--", label="Prop (rk4)", linewidth=1.8)

    ax.scatter([r0_np[0]], [r0_np[1]], [r0_np[2]], s=60, marker="o", label="r0")
    ax.scatter([rf_np[0]], [rf_np[1]], [rf_np[2]], s=90, marker="*", label="rf")

    cone = cfg["constraints"]["cone"]
    gamma_deg = float(cone["gamma_max_deg"])
    n_hat = np.array(cone["n_hat"], dtype=float)
    tc_s = float(cone["activation"]["tc_s"])
    show_cone = args.show_cone_always or (t_prop_np[-1] >= tc_s)
    if show_cone:
        X, Y, Z = make_cone_surface(
            apex=rf_np,
            axis=n_hat,
            gamma_deg=gamma_deg,
            height=float(args.cone_height),
        )
        ax.plot_surface(X, Y, Z, alpha=0.15, linewidth=0.0)

        a_unit = n_hat / (np.linalg.norm(n_hat) + 1e-15)
        a_end = rf_np + a_unit * float(args.cone_height)
        ax.plot(
            [rf_np[0], a_end[0]],
            [rf_np[1], a_end[1]],
            [rf_np[2], a_end[2]],
            linewidth=1.5,
            label="cone axis",
        )

    ax.set_title(f"CE(trained) vs Propagate  (u_mode={args.u_mode})")
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    ax.legend()
    set_axes_equal(ax)
    save_fig(fig1, outdir / f"fig1_traj3d_{suffix}.png")

    # =========================
    # fig2: r/v/a/u
    # =========================
    fig2, axes2 = plt.subplots(3, 4, figsize=(15, 8), sharex=True)
    comps = ["x", "y", "z"]
    cols = ["r", "v", "a", "u"]

    for i in range(3):
        axes2[i, 0].plot(t_train_np, r_ce[:, i], linewidth=2.0, label="CE(trained)" if (i == 0) else None)
        axes2[i, 0].plot(t_prop_np,  r_pr[:, i], "--", linewidth=1.6, label="Prop" if (i == 0) else None)
        axes2[i, 0].scatter(t_train_np[0],  r0_np[i], marker="o", s=40, zorder=5)
        axes2[i, 0].scatter(t_train_np[-1], rf_np[i], marker="*", s=60, zorder=5)
        axes2[i, 0].set_ylabel(comps[i])
        if i == 0: axes2[i, 0].set_title(cols[0])
        axes2[i, 0].grid(True, alpha=0.3)

        axes2[i, 1].plot(t_train_np, v_ce[:, i], linewidth=2.0)
        axes2[i, 1].plot(t_prop_np,  v_pr[:, i], "--", linewidth=1.6)
        axes2[i, 1].scatter(t_train_np[0],  v0_np[i], marker="o", s=40, zorder=5)
        axes2[i, 1].scatter(t_train_np[-1], vf_np[i], marker="*", s=60, zorder=5)
        if i == 0: axes2[i, 1].set_title(cols[1])
        axes2[i, 1].grid(True, alpha=0.3)

        axes2[i, 2].plot(t_train_np, a_ce[:, i], linewidth=2.0)
        axes2[i, 2].plot(t_prop_np,  a_pr[:, i], "--", linewidth=1.6)
        if i == 0: axes2[i, 2].set_title(cols[2])
        axes2[i, 2].grid(True, alpha=0.3)

        axes2[i, 3].plot(t_train_np, u_ce_np[:, i], linewidth=2.0)
        axes2[i, 3].plot(t_prop_np,  u_pr_np[:, i], "--", linewidth=1.2)
        if i == 0: axes2[i, 3].set_title(cols[3])
        axes2[i, 3].grid(True, alpha=0.3)

    for j in range(4):
        axes2[2, j].set_xlabel("t [s]")

    axes2[0, 0].legend(loc="best")
    fig2.suptitle("rows: x/y/z | cols: r/v/a/u", y=0.98)
    fig2.tight_layout()
    save_fig(fig2, outdir / f"fig2_rvau_{suffix}.png")

    # =========================
    # fig3: costates
    # =========================
    fig3, axes3 = plt.subplots(3, 4, figsize=(15, 8), sharex=True)
    cols3 = ["lam_r", "lam_r_dot", "lam_v", "lam_v_dot"]

    for i in range(3):
        axes3[i, 0].plot(t_train_np, lam_r_ce[:, i], linewidth=2.0, label="CE(trained)" if (i == 0) else None)
        axes3[i, 0].set_ylabel(comps[i])
        if i == 0: axes3[i, 0].set_title(cols3[0])
        axes3[i, 0].grid(True, alpha=0.3)

        axes3[i, 1].plot(t_train_np, lam_r_dot_ce[:, i], linewidth=2.0)
        if i == 0: axes3[i, 1].set_title(cols3[1])
        axes3[i, 1].grid(True, alpha=0.3)

        axes3[i, 2].plot(t_train_np, lam_v_ce[:, i], linewidth=2.0)
        if i == 0: axes3[i, 2].set_title(cols3[2])
        axes3[i, 2].grid(True, alpha=0.3)

        axes3[i, 3].plot(t_train_np, lam_v_dot_ce[:, i], linewidth=2.0)
        if i == 0: axes3[i, 3].set_title(cols3[3])
        axes3[i, 3].grid(True, alpha=0.3)

    for j in range(4):
        axes3[2, j].set_xlabel("t [s]")

    axes3[0, 0].legend(loc="best")
    fig3.suptitle("rows: x/y/z | cols: lam_r/lam_r_dot/lam_v/lam_v_dot", y=0.98)
    fig3.tight_layout()
    save_fig(fig3, outdir / f"fig3_costates_{suffix}.png")

    # =========================
    # fig4: mu_gs / mu_a
    # =========================
    fig4, axes4 = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes4[0].plot(t_train_np, mu_gs_ce, linewidth=2.0, label="mu_gs (trained)")
    axes4[0].set_xlabel("t [s]")
    axes4[0].set_ylabel("mu_gs")
    axes4[0].grid(True, alpha=0.3)
    axes4[0].legend(loc="best")

    axes4[1].plot(t_train_np, mu_a_ce, linewidth=2.0, label="mu_a (trained)")
    axes4[1].set_xlabel("t [s]")
    axes4[1].set_ylabel("mu_a")
    axes4[1].grid(True, alpha=0.3)
    axes4[1].legend(loc="best")

    fig4.suptitle("rows: mu_gs/mu_a", y=0.98)
    fig4.tight_layout()
    save_fig(fig4, outdir / f"fig4_multipliers_{suffix}.png")

    # =========================
    # fig5: a_c
    # =========================
    fig5 = plt.figure(figsize=(10, 3))
    ax5 = fig5.add_subplot(111)
    ax5.plot(t_train_np, ac_ce, linewidth=2.0, label="a_c from CE(costate)")
    ax5.plot(t_prop_np, ac_pr, "--", linewidth=1.6, label="a_c (prop, ||u_prop||)")
    ax5.set_title("a_c(t) (scalar)")
    ax5.set_xlabel("t [s]")
    ax5.set_ylabel("a_c [km/s^2]")
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc="best")
    fig5.tight_layout()
    save_fig(fig5, outdir / f"fig5_ac_{suffix}.png")

    # =========================
    # fig6: 2D XY trajectory (square figure, fixed x/y limits, cone triangle)
    # =========================
    fig6, ax6 = plt.subplots(figsize=(7, 7))  # square canvas

    # trajectories
    ax6.plot(r_ce[:, 0], r_ce[:, 1], linewidth=2.0, label="CE (trained)")
    ax6.plot(r_pr[:, 0], r_pr[:, 1], "--", linewidth=1.6, label="Prop (rk4)")

    # boundary points
    ax6.scatter(r0_np[0], r0_np[1], marker="o", s=70, label="r0")
    ax6.scatter(rf_np[0], rf_np[1], marker="*", s=110, label="rf")

    # -------------------------
    # Cone as 2D filled triangle in XY
    # -------------------------
    cone = cfg["constraints"]["cone"]
    gamma_deg = float(cone["gamma_max_deg"])
    gamma = np.deg2rad(gamma_deg)

    apex_xy = np.array([rf_np[0], rf_np[1]], dtype=float)

    # axis direction in XY
    axis_xy = np.array([cone["n_hat"][0], cone["n_hat"][1]], dtype=float)
    axis_xy_norm = np.linalg.norm(axis_xy)

    if axis_xy_norm > 1e-12:
        axis_xy = axis_xy / axis_xy_norm
        perp_xy = np.array([-axis_xy[1], axis_xy[0]], dtype=float)

        h = float(args.cone_height)
        base_center = apex_xy + axis_xy * h
        half_width = h * np.tan(gamma)

        left_pt = base_center + perp_xy * half_width
        right_pt = base_center - perp_xy * half_width

        # ---- fill triangle ----
        tri_x = [apex_xy[0], left_pt[0], right_pt[0]]
        tri_y = [apex_xy[1], left_pt[1], right_pt[1]]
        ax6.fill(tri_x, tri_y, alpha=0.18, label="cone feasible region")  # no edgecolor -> no base outline

        # ---- dashed sides only (no base) ----
        ax6.plot(
            [apex_xy[0], left_pt[0]],
            [apex_xy[1], left_pt[1]],
            linestyle="--",
            linewidth=1.2,
            label="cone edges"
        )
        ax6.plot(
            [apex_xy[0], right_pt[0]],
            [apex_xy[1], right_pt[1]],
            linestyle="--",
            linewidth=1.2
        )

        # ---- axis (thinner) ----
        ax6.plot(
            [apex_xy[0], base_center[0]],
            [apex_xy[1], base_center[1]],
            linestyle="--",
            linewidth=0.8,   # thinner
            label="cone axis"
        )

    else:
        print("[warn] n_hat has near-zero XY component; cannot draw cone.")
    # -------------------------
    # Fixed plot limits (as requested)
    # -------------------------
    ax6.set_xlim(-0.2, 0.1)
    ax6.set_ylim(-1.3, 0.0)

    # IMPORTANT: do NOT force equal aspect; allow different x/y scales
    # ax6.set_aspect("auto")  # (default)

    ax6.set_title("XY Trajectory Projection (fixed limits)")
    ax6.set_xlabel("x [km]")
    ax6.set_ylabel("y [km]")
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc="best")

    save_fig(fig6, outdir / f"fig6_xy_{suffix}.png")

    if args.show:
        # if you want to see them too, re-open by not closing; easiest is to just not show by default.
        # Here we just inform where files are saved.
        print("[info] Figures saved. Re-run with --show if you want interactive view (optional).")
        # If you really want interactive display too, comment out plt.close in save_fig and call plt.show() here.

    print("Done.")


if __name__ == "__main__":
    main()