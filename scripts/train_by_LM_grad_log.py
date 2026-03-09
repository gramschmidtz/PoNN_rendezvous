# scripts/train_by_LM.py
from __future__ import annotations

from csv import writer
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
import math
import datetime
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ponn.ce import CEModel
from src.ponn.dynamics import SSOrbitParams, compute_ss_coeffs
from src.ponn.init_guess import TrainParams, init_guess
from src.ponn.constraint import ConeConstraint, ConeActivation, ControlConstraint
from src.ponn.losses import RendezvousLossWeights, RendezvousLosses
from src.ponn.problem import RendezvousProblem
from src.ponn.lm_solver import LMSolverConfig, solve_lm


# -------------------------
# YAML / helpers
# -------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _to_vec3(x, *, device, dtype) -> torch.Tensor:
    t = x if torch.is_tensor(x) else torch.tensor(x)
    return t.to(device=device, dtype=dtype).reshape(3)


def make_orbit(cfg: Dict[str, Any]) -> SSOrbitParams:
    o = cfg["orbit"]
    it_rad = float(o["it_deg"]) * (3.141592653589793 / 180.0)
    return SSOrbitParams(
        Re_km=float(o["Re_km"]),
        Rt_km=float(o["Rt_km"]),
        it_rad=it_rad,
        mu_km3_s2=float(o["mu_km3_s2"]),
        J2=float(o["J2"]),
    )


def make_time_grid(cfg: Dict[str, Any], *, device, dtype) -> torch.Tensor:
    t0 = float(cfg["time"]["t0"])
    tf = float(cfg["time"]["tf"])
    n_dis = int(cfg["discretization"]["n_dis"])
    return torch.linspace(t0, tf, n_dis, device=device, dtype=dtype)


def make_constraints(
    cfg: Dict[str, Any],
    *,
    rf: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = 1e-12,
) -> Tuple[ConeConstraint, ControlConstraint]:
    c = cfg["constraints"]
    cone_cfg = c["cone"]
    act_cfg = cone_cfg["activation"]

    cone_cstr = ConeConstraint(
        rf=rf.to(device=device, dtype=dtype).reshape(3),
        n_hat=_to_vec3(cone_cfg["n_hat"], device=device, dtype=dtype),
        gamma_max_deg=float(cone_cfg["gamma_max_deg"]),
        activation=ConeActivation(
            mode=act_cfg.get("mode", "time"),
            t_c=float(act_cfg["tc_s"]),
            k=float(act_cfg["k"]),
        ),
        eps=eps,
    )

    ctrl_cstr = ControlConstraint(
        a_c_max=float(c["control_max_km_s2"]),
        eps=eps,
    )

    return cone_cstr, ctrl_cstr


# -------------------------
# 11 losses from packed residual (11N,)
# -------------------------

def _cost_half_sumsq(x: torch.Tensor) -> float:
    """0.5 * sum(x^2)"""
    x = x.reshape(-1)
    return float((0.5 * (x * x).sum()).detach().cpu().item())


def split_pack_11N(resid: torch.Tensor, N: int) -> Dict[str, torch.Tensor]:
    """
    resid: (11N,) = [Lv(3N), Llam_r(3N), Llam_v(3N), Lgs(N), La(N)]
    returns tensors:
      Lv: (N,3), Llam_r: (N,3), Llam_v: (N,3), Lgs: (N,), La: (N,)
    """
    r = resid.reshape(-1)
    i = 0
    Lv = r[i:i + 3 * N].reshape(N, 3); i += 3 * N
    Llam_r = r[i:i + 3 * N].reshape(N, 3); i += 3 * N
    Llam_v = r[i:i + 3 * N].reshape(N, 3); i += 3 * N
    Lgs = r[i:i + N].reshape(N); i += N
    La = r[i:i + N].reshape(N); i += N
    return {"Lv": Lv, "Llam_r": Llam_r, "Llam_v": Llam_v, "Lgs": Lgs, "La": La}


def split_x_groups(g: torch.Tensor, L: int, N: int) -> Dict[str, torch.Tensor]:
    """
    g: (9L + 2N,)  pack order:
      [beta_r(3L), beta_lam_r(3L), beta_lam_v(3L), mu_gs(N), mu_a(N)]
    """
    g = g.reshape(-1)
    n_beta = 3 * L
    i0 = 0
    i1 = i0 + n_beta
    i2 = i1 + n_beta
    i3 = i2 + n_beta
    i4 = i3 + N
    i5 = i4 + N

    if g.numel() != i5:
        raise RuntimeError(f"trainvector length mismatch: got {g.numel()} expected {i5}")

    return {
        "beta_r": g[i0:i1],
        "beta_lam_r": g[i1:i2],
        "beta_lam_v": g[i2:i3],
        "mu_gs": g[i3:i4],
        "mu_a": g[i4:i5],
        "total": g,
    }


def l2norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.reshape(-1), ord=2).detach().cpu().item())

def infnorm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.reshape(-1), ord=float("inf")).detach().cpu().item())


def log_cost11(writer: SummaryWriter, it: int, resid: torch.Tensor, N: int) -> None:
    blocks = split_pack_11N(resid, N)

    Lv = blocks["Lv"]
    Lr = blocks["Llam_r"]
    Lv2 = blocks["Llam_v"]
    Lgs = blocks["Lgs"]
    La = blocks["La"]

    writer.add_scalar("cost11/Lv_x", _cost_half_sumsq(Lv[:, 0]), it)
    writer.add_scalar("cost11/Lv_y", _cost_half_sumsq(Lv[:, 1]), it)
    writer.add_scalar("cost11/Lv_z", _cost_half_sumsq(Lv[:, 2]), it)

    writer.add_scalar("cost11/Llam_r_x", _cost_half_sumsq(Lr[:, 0]), it)
    writer.add_scalar("cost11/Llam_r_y", _cost_half_sumsq(Lr[:, 1]), it)
    writer.add_scalar("cost11/Llam_r_z", _cost_half_sumsq(Lr[:, 2]), it)

    writer.add_scalar("cost11/Llam_v_x", _cost_half_sumsq(Lv2[:, 0]), it)
    writer.add_scalar("cost11/Llam_v_y", _cost_half_sumsq(Lv2[:, 1]), it)
    writer.add_scalar("cost11/Llam_v_z", _cost_half_sumsq(Lv2[:, 2]), it)

    writer.add_scalar("cost11/Lgs", _cost_half_sumsq(Lgs), it)
    writer.add_scalar("cost11/La", _cost_half_sumsq(La), it)


def log_cost5(writer: SummaryWriter, it: int, resid: torch.Tensor, N: int) -> None:
    blocks = split_pack_11N(resid, N)

    writer.add_scalar("cost5/Lv", _cost_half_sumsq(blocks["Lv"]), it)
    writer.add_scalar("cost5/Llam_r", _cost_half_sumsq(blocks["Llam_r"]), it)
    writer.add_scalar("cost5/Llam_v", _cost_half_sumsq(blocks["Llam_v"]), it)
    writer.add_scalar("cost5/Lgs", _cost_half_sumsq(blocks["Lgs"]), it)
    writer.add_scalar("cost5/La", _cost_half_sumsq(blocks["La"]), it)


def log_grad5(writer: SummaryWriter, it: int, problem: RendezvousProblem, x: torch.Tensor, N: int) -> None:
    x_req = x.detach().clone().requires_grad_(True)
    resid = problem(x_req)

    blocks = split_pack_11N(resid, N)

    comps = [
        ("grad5/Lv_l2", blocks["Lv"]),
        ("grad5/Llam_r_l2", blocks["Llam_r"]),
        ("grad5/Llam_v_l2", blocks["Llam_v"]),
        ("grad5/Lgs_l2", blocks["Lgs"]),
        ("grad5/La_l2", blocks["La"]),
    ]

    for name, r_blk in comps:
        cost_blk = 0.5 * torch.sum(r_blk.reshape(-1) ** 2)
        g = torch.autograd.grad(cost_blk, x_req, retain_graph=True)[0]
        g_l2 = torch.linalg.vector_norm(g, ord=2).detach().cpu().item()
        writer.add_scalar(name, g_l2, it)


def log_grad11_components_to_tb(
    writer: SummaryWriter,
    step: int,
    problem: RendezvousProblem,
    x: torch.Tensor,
    L: int,
    N: int,
    *,
    max_per_group: int | None = None,
    stride: int = 1,
) -> None:
    x_req = x.detach().clone().requires_grad_(True)
    resid = problem(x_req)
    blocks = split_pack_11N(resid, N)

    comps = [
        ("Lv_x",     blocks["Lv"][:, 0]),
        ("Lv_y",     blocks["Lv"][:, 1]),
        ("Lv_z",     blocks["Lv"][:, 2]),
        ("Llam_r_x", blocks["Llam_r"][:, 0]),
        ("Llam_r_y", blocks["Llam_r"][:, 1]),
        ("Llam_r_z", blocks["Llam_r"][:, 2]),
        ("Llam_v_x", blocks["Llam_v"][:, 0]),
        ("Llam_v_y", blocks["Llam_v"][:, 1]),
        ("Llam_v_z", blocks["Llam_v"][:, 2]),
        ("Lgs",      blocks["Lgs"]),
        ("La",       blocks["La"]),
    ]

    for loss_name, r_part in comps:
        cost_i = 0.5 * torch.sum(r_part.reshape(-1) ** 2)
        g_i = torch.autograd.grad(cost_i, x_req, retain_graph=True, create_graph=False)[0].reshape(-1)

        # ✅ 여기서 "각 컴포넌트 값"을 로깅
        log_grad_vector_by_group_components(
            writer,
            step,
            tag_prefix=f"grad11vec/{loss_name}",
            g=g_i,
            L=L,
            N=N,
            max_per_group=max_per_group,
            stride=stride,
        )


def log_grad_cancellation(
    writer: SummaryWriter,
    step: int,
    problem: RendezvousProblem,
    x: torch.Tensor,
    L: int,
    N: int,
) -> None:
    """
    cancellation 지표:
      g_total = d(0.5||resid||^2)/dx
      g_sum   = sum_i d(0.5||resid_i||^2)/dx

    로깅:
      cancel/total_grad_l2
      cancel/sum_component_norms_l2
      cancel/ratio_total_over_sum  (= ||g_total|| / sum_i ||g_i|| )
      cancel/diff_total_minus_sum_l2  (= ||g_total - g_sum||, sanity check)
    """
    x_req = x.detach().clone().requires_grad_(True)
    resid = problem(x_req)  # (11N,)
    blocks = split_pack_11N(resid, N)

    # total grad
    cost_total = 0.5 * torch.sum(resid.reshape(-1) ** 2)
    g_total = torch.autograd.grad(cost_total, x_req, retain_graph=True, create_graph=False)[0].reshape(-1)

    # component grads (11개 합)
    comps = [
        blocks["Lv"][:, 0], blocks["Lv"][:, 1], blocks["Lv"][:, 2],
        blocks["Llam_r"][:, 0], blocks["Llam_r"][:, 1], blocks["Llam_r"][:, 2],
        blocks["Llam_v"][:, 0], blocks["Llam_v"][:, 1], blocks["Llam_v"][:, 2],
        blocks["Lgs"], blocks["La"],
    ]

    g_sum = torch.zeros_like(g_total)
    sum_norms = 0.0

    for r_part in comps:
        cost_i = 0.5 * torch.sum(r_part.reshape(-1) ** 2)
        g_i = torch.autograd.grad(cost_i, x_req, retain_graph=True, create_graph=False)[0].reshape(-1)
        g_sum = g_sum + g_i
        sum_norms += l2norm(g_i)

    # cancellation metrics
    total_norm = l2norm(g_total)
    diff_norm  = l2norm(g_total - g_sum)  # 이건 거의 0이어야 정상(검증용)
    ratio = float(total_norm / (sum_norms + 1e-30))

    writer.add_scalar("cancel/total_grad_l2", total_norm, step)
    writer.add_scalar("cancel/sum_component_norms_l2", sum_norms, step)
    writer.add_scalar("cancel/ratio_total_over_sum", ratio, step)
    writer.add_scalar("cancel/diff_total_minus_sum_l2", diff_norm, step)

    # (옵션) 그룹별 total grad도 같이 보면 더 직관적
    gs = split_x_groups(g_total, L, N)
    writer.add_scalar("cancel/by_group/beta_r_l2",     l2norm(gs["beta_r"]), step)
    writer.add_scalar("cancel/by_group/beta_lam_r_l2", l2norm(gs["beta_lam_r"]), step)
    writer.add_scalar("cancel/by_group/beta_lam_v_l2", l2norm(gs["beta_lam_v"]), step)
    writer.add_scalar("cancel/by_group/mu_gs_l2",      l2norm(gs["mu_gs"]), step)
    writer.add_scalar("cancel/by_group/mu_a_l2",       l2norm(gs["mu_a"]), step)


def log_vector_components(
    writer: SummaryWriter,
    step: int,
    tag_prefix: str,
    v: torch.Tensor,
    *,
    max_elems: int | None = None,
    stride: int = 1,
) -> None:
    """
    v의 각 컴포넌트를 TensorBoard scalar로 로깅.
    tag: {tag_prefix}/{idx:04d}
    """
    vv = v.detach().reshape(-1).cpu()
    n = vv.numel()
    if max_elems is None:
        max_elems = n

    # 안전장치: 너무 많으면 max_elems까지만
    end = min(n, max_elems)

    for i in range(0, end, stride):
        writer.add_scalar(f"{tag_prefix}/{i:04d}", float(vv[i].item()), step)


def log_grad_vector_by_group_components(
    writer: SummaryWriter,
    step: int,
    tag_prefix: str,
    g: torch.Tensor,
    L: int,
    N: int,
    *,
    max_per_group: int | None = None,
    stride: int = 1,
) -> None:
    gs = split_x_groups(g, L, N)

    # 그룹별로 "컴포넌트 값"을 그대로 로깅
    log_vector_components(writer, step, f"{tag_prefix}/beta_r",     gs["beta_r"],     max_elems=max_per_group, stride=stride)
    log_vector_components(writer, step, f"{tag_prefix}/beta_lam_r", gs["beta_lam_r"], max_elems=max_per_group, stride=stride)
    log_vector_components(writer, step, f"{tag_prefix}/beta_lam_v", gs["beta_lam_v"], max_elems=max_per_group, stride=stride)
    log_vector_components(writer, step, f"{tag_prefix}/mu_gs",      gs["mu_gs"],      max_elems=max_per_group, stride=stride)
    log_vector_components(writer, step, f"{tag_prefix}/mu_a",       gs["mu_a"],       max_elems=max_per_group, stride=stride)

    # (옵션) 전체 벡터도 찍고 싶으면
    # log_vector_components(writer, step, f"{tag_prefix}/total", gs["total"], max_elems=max_per_group, stride=stride)


def log_total_grad_components_to_tb(
    writer: SummaryWriter,
    step: int,
    problem: RendezvousProblem,
    x: torch.Tensor,
    L: int,
    N: int,
    *,
    max_per_group: int | None = None,
    stride: int = 1,
) -> None:
    x_req = x.detach().clone().requires_grad_(True)
    resid = problem(x_req)
    cost_total = 0.5 * torch.sum(resid.reshape(-1) ** 2)
    g_total = torch.autograd.grad(cost_total, x_req, retain_graph=False, create_graph=False)[0].reshape(-1)

    log_grad_vector_by_group_components(
        writer,
        step,
        tag_prefix="grad_total_vec",
        g=g_total,
        L=L,
        N=N,
        max_per_group=max_per_group,
        stride=stride,
    )


# -------------------------
# main
# -------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/rendezvous_iv.yaml")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=None, choices=[None, "float32", "float64"])
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--log_every", type=int, default=1, help="TensorBoard logging frequency (iterations)")
    args = ap.parse_args()

    cfg = load_yaml(Path(ROOT / args.config))

    device_str = args.device or cfg.get("train", {}).get("device", "cpu")
    dtype_str = args.dtype or cfg.get("train", {}).get("dtype", "float64")
    device = torch.device(device_str)
    dtype = torch.float64 if dtype_str == "float64" else torch.float32

    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)

    # coeffs + model
    orbit = make_orbit(cfg)
    coeffs = compute_ss_coeffs(orbit, device=device, dtype=dtype)
    model = CEModel.from_yaml(cfg, seed=seed, device=device, dtype=dtype)

    # boundary
    r0 = _to_vec3(cfg["boundary"]["r0_km"], device=device, dtype=dtype)
    v0 = _to_vec3(cfg["boundary"]["v0_km_s"], device=device, dtype=dtype)
    rf = _to_vec3(cfg["boundary"]["rf_km"], device=device, dtype=dtype)
    vf = _to_vec3(cfg["boundary"]["vf_km_s"], device=device, dtype=dtype)

    # time grid
    t = make_time_grid(cfg, device=device, dtype=dtype)
    t0 = float(cfg["time"]["t0"])
    tf = float(cfg["time"]["tf"])

    # init guess
    ig = cfg.get("init_guess", {})
    mode = ig.get("method", "linear")
    ridge = float(ig.get("ridge", 1e-8))
    lam_r_const = float(ig.get("lam_r_const", 0.0))
    mu_a_init = float(ig.get("mu_a_init", 1e-3))
    mu_gs_init = float(ig.get("mu_gs_init", 1.0))

    guess = init_guess(
        model,
        t,
        coeffs=coeffs,
        r0=r0, v0=v0, rf=rf, vf=vf,
        mode=mode,
        lam_r_const=lam_r_const,
        mu_a_init=mu_a_init,
        mu_gs_init=mu_gs_init,
        ridge=ridge,
    )
    x0 = guess.pack().to(device=device, dtype=dtype)

    # constraints
    cone_cstr, ctrl_cstr = make_constraints(cfg, rf=rf, device=device, dtype=dtype, eps=1e-12)

    weights = RendezvousLossWeights(
        w_v=float(cfg["loss_weights"]["w_v"]),
        w_a=float(cfg["loss_weights"]["w_a"]),
        w_gs=float(cfg["loss_weights"]["w_gs"]),
        w_lam_r=float(cfg["loss_weights"]["w_lam_r"]),
        w_lam_v=float(cfg["loss_weights"]["w_lam_v"]),
    )

    # loss + problem
    L = model.basis.w.numel()
    N = t.numel()

    loss = RendezvousLosses(
        coeffs=coeffs,
        weights=weights,
        cone_cstr=cone_cstr,
        ctrl_cstr=ctrl_cstr,
        t0=t0,
        tf=tf,
    )

    problem = RendezvousProblem(
        model=model,
        t=t,
        r0=r0, v0=v0, rf=rf, vf=vf,
        loss=loss,
        L=L,
        N=N,
        device=device,
        dtype=dtype,
    )

    run = cfg.get("run", {})
    base_out_dir = Path(ROOT / run.get("out_dir", "runs"))
    exp_name = run.get("name", "rendezvous_iv_lm")

    # run_id: timestamp + short uuid (collision-safe)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    # final structure:
    # runs/<exp_name>/<run_id>/
    out_path = base_out_dir / exp_name / run_id
    out_path.mkdir(parents=True, exist_ok=True)

    # TensorBoard writer:
    # runs/<exp_name>/<run_id>/tb
    tb_dir = out_path / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    # (optional) write a small text file to know which run this is
    (out_path / "run_id.txt").write_text(run_id + "\n", encoding="utf-8")

    # (optional) keep a "latest" pointer (symlink if possible; fallback to a text file)
    latest = base_out_dir / exp_name / "LATEST"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(out_path, target_is_directory=True)
    except Exception:
        # windows 권한/설정 때문에 symlink가 실패할 수 있음
        (base_out_dir / exp_name / "LATEST_PATH.txt").write_text(str(out_path) + "\n", encoding="utf-8")

    print(f"[train] run dir: {out_path}")
    print(f"[train] tb dir : {tb_dir}")

    # LM config
    opt = cfg.get("optimizer", {})
    lm_cfg = LMSolverConfig(
        damping0=float(opt.get("damping0", 1e-2)),
        damping_min=float(opt.get("damping_min", 1e-20)),
        damping_max=float(opt.get("damping_max", 1e20)),
        damp_increase=float(opt.get("damp_increase", 10.0)),
        damp_decrease=float(opt.get("damp_decrease", 0.3)),
        max_iters=int(opt.get("max_iters", 200)),
        grad_inf_tol=float(opt.get("grad_inf_tol", 1e-10)),
        step_inf_tol=float(opt.get("step_inf_tol", 1e-12)),
        verbose=True,
    )

    # callback: realtime tensorboard logging
    tb_step = 0

    def tb_callback(it: int, x: torch.Tensor, l: torch.Tensor, cost: float, g_inf: float, lam: float) -> None:
        nonlocal tb_step
        if (it % int(args.log_every)) != 0:
            return

        # 전체 LM cost
        writer.add_scalar("lm/cost", cost, tb_step)

        log_cost5(writer, tb_step, l, N)
        log_cost11(writer, tb_step, l, N)

        # grad
        if it % 10 == 0:
            log_grad5(writer, tb_step, problem, x, N)
            log_grad11_components_to_tb(writer, tb_step, problem, x, L, N, max_per_group=None, stride=1)
            log_total_grad_components_to_tb(writer, tb_step, problem, x, L, N, max_per_group=None, stride=1)

        writer.flush()
        tb_step += 1

    # solve
    x_best, info = solve_lm(x0, problem, lm_cfg, callback=tb_callback)

    writer.close()
    print(f"[train] tensorboard logdir: {base_out_dir / exp_name}")
    print(f"Run: tensorboard --logdir {base_out_dir / exp_name}")

    # unpack + save checkpoint
    tp_best = TrainParams.unpack(x_best, L, N)
    betas_best = tp_best.betas
    mu_gs_best = tp_best.mu_gs
    mu_a_best = tp_best.mu_a

    ckpt = {
        "config_path": str(args.config),
        "seed": seed,
        "device": device_str,
        "dtype": dtype_str,
        "betas": {
            "beta_r": betas_best.beta_r.detach().cpu(),
            "beta_lam_r": betas_best.beta_lam_r.detach().cpu(),
            "beta_lam_v": betas_best.beta_lam_v.detach().cpu(),
        },
        "mu_a": mu_a_best.detach().cpu(),
        "mu_gs": mu_gs_best.detach().cpu(),
        "lm_info": {
            "best_cost": info["best_cost"],
            "best_it": info["best_it"],
            "final_cost": info["final_cost"],
            "final_it": info["final_it"],
        },
    }
    torch.save(ckpt, out_path / "checkpoint.pt")
    print(f"[train] saved: {out_path / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()