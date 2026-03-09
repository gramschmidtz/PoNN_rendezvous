# scripts/plot_train_result.py
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ----------------------------
# TB event reader
# ----------------------------
def load_scalars_from_event(event_file: Path) -> Tuple[List[str], Dict[str, Tuple[List[int], List[float]]]]:
    acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    acc.Reload()

    tags = acc.Tags().get("scalars", [])
    scalars: Dict[str, Tuple[List[int], List[float]]] = {}

    for tag in tags:
        events = acc.Scalars(tag)
        steps = [int(e.step) for e in events]
        values = [float(e.value) for e in events]
        scalars[tag] = (steps, values)

    return tags, scalars


# ----------------------------
# Path helpers (new run layout)
# ----------------------------
EVENT_PREFIX = "events.out.tfevents."


def find_event_files(tb_dir: Path) -> List[Path]:
    if not tb_dir.exists():
        return []
    return sorted(tb_dir.glob(f"{EVENT_PREFIX}*"), key=lambda p: p.stat().st_mtime)


def pick_event_files(tb_dir: Path, policy: str) -> List[Path]:
    evs = find_event_files(tb_dir)
    if not evs:
        raise FileNotFoundError(f"No event files found in: {tb_dir}")

    if policy == "latest":
        return [evs[-1]]
    if policy == "all":
        return evs

    raise ValueError(f"Unknown policy: {policy}")


def is_run_dir(p: Path) -> bool:
    return (p / "tb").is_dir()


def list_run_dirs(exp_dir: Path) -> List[Path]:
    # runs/<exp>/<run_id>/ ... where run_dir has tb/
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


# ----------------------------
# Tag helpers
# ----------------------------
def safe_filename(name: str) -> str:
    return re.sub(r"[^\w\-_\.]+", "_", name).strip("_")


def clip_for_log(values: List[float], eps: float = 1e-12) -> List[float]:
    return [max(float(v), eps) for v in values]


def select_tags(tags: List[str], *, regex: Optional[str] = None, contains: Optional[str] = None) -> List[str]:
    out = tags
    if regex is not None:
        r = re.compile(regex)
        out = [t for t in out if r.search(t)]
    if contains is not None:
        out = [t for t in out if contains in t]
    return sorted(out)


# ----------------------------
# Plot helpers
# ----------------------------
def plot_series(ax, tag: str, steps: List[int], values: List[float], *, ylog: bool):
    if ylog:
        values = clip_for_log(values)
        ax.set_yscale("log")
    ax.plot(steps, values)
    ax.set_title(tag, fontsize=9)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)


def save_fig(fig, outpath: Path, dpi: int = 300):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outpath}")


# ----------------------------
# Figure #1: 5x2 (loss5 vs grad)
# ----------------------------
def plot_loss5_grad_grid(tags, scalars, outdir, *, ylog):
    order = ["Lv", "Llam_r", "Llam_v", "Lgs", "La"]
    fig, axes = plt.subplots(5, 2, figsize=(16, 18))

    for i, key in enumerate(order):
        # cost5 (left)
        loss_tag = next((t for t in tags if t.startswith("cost5/") and key in t), None)
        axL = axes[i, 0]
        if loss_tag:
            steps, values = scalars[loss_tag]
            plot_series(axL, loss_tag, steps, values, ylog=ylog)
        else:
            axL.axis("off")

        # grad (right)
        grad_tag = next((t for t in tags if t.startswith("grad5/") and key in t), None)
        axR = axes[i, 1]
        if grad_tag:
            steps, values = scalars[grad_tag]
            plot_series(axR, grad_tag, steps, values, ylog=ylog)
        else:
            axR.axis("off")

    save_fig(fig, outdir / "fig1_loss5_vs_grad_5x2.png")


# ----------------------------
# Figure #2: cost11 11개 (3x4)
# ----------------------------
def plot_cost11_grid(tags, scalars, outdir, *, ylog):
    rows = ["x", "y", "z"]
    cols_main = ["Lv", "Llam_r", "Llam_v"]

    fig, axes = plt.subplots(3, 4, figsize=(18, 10))

    # main 3 cols: (Lv, Llam_r, Llam_v) x/y/z
    for i, axis_name in enumerate(rows):
        for j, loss_name in enumerate(cols_main):
            tag = next((t for t in tags if t.startswith("cost11/") and loss_name in t and axis_name in t), None)
            ax = axes[i, j]
            if tag:
                steps, values = scalars[tag]
                plot_series(ax, tag, steps, values, ylog=ylog)
            else:
                ax.axis("off")

    # last col: Lgs and La (scalar)
    tag_gs = next((t for t in tags if t == "cost11/Lgs"), None)
    tag_a = next((t for t in tags if t == "cost11/La"), None)

    ax = axes[0, 3]
    if tag_gs:
        steps, values = scalars[tag_gs]
        plot_series(ax, tag_gs, steps, values, ylog=ylog)
    else:
        ax.axis("off")

    ax = axes[1, 3]
    if tag_a:
        steps, values = scalars[tag_a]
        plot_series(ax, tag_a, steps, values, ylog=ylog)
    else:
        ax.axis("off")

    axes[2, 3].axis("off")

    save_fig(fig, outdir / "fig2_cost11_grid_3x4.png")


# ----------------------------
# Figure #3: LM cost 단일 plot
# ----------------------------
def plot_lm_cost(tags, scalars, outdir: Path, *, ylog: bool):
    # your script writes: writer.add_scalar("lm/cost", ...)
    t = "lm/cost" if "lm/cost" in tags else None
    if t is None:
        # fallback heuristic
        cand = [x for x in tags if ("lm" in x.lower() and "cost" in x.lower())]
        t = cand[0] if cand else None

    if t is None:
        print("⚠ LM cost tag not found.")
        return

    steps, values = scalars[t]
    fig = plt.figure(figsize=(8, 5))
    ax = fig.gca()
    plot_series(ax, t, steps, values, ylog=ylog)
    save_fig(fig, outdir / "fig3_lm_cost.png")


def process_one_event(event_file: Path, outdir: Path, *, ylog: bool) -> None:
    print(f"[event]  {event_file}")
    tags, scalars = load_scalars_from_event(event_file)
    print(f"[scalars] {len(tags)} tag(s) found")

    plot_loss5_grad_grid(tags, scalars, outdir, ylog=ylog)
    plot_cost11_grid(tags, scalars, outdir, ylog=ylog)
    plot_lm_cost(tags, scalars, outdir, ylog=ylog)


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    # New interface:
    ap.add_argument("--logdir", type=str, default=None,
                    help="Experiment dir: runs/<exp_name> (contains run_id subdirs).")
    ap.add_argument("--run-dir", type=str, default=None,
                    help="Run dir: runs/<exp_name>/<run_id> (contains tb/).")
    ap.add_argument("--run-id", type=str, default=None,
                    help="Pick a specific run_id under --logdir. If omitted, picks latest.")

    ap.add_argument("--event-policy", choices=["latest", "all"], default="latest",
                    help="If tb has multiple event files, pick latest or process all.")

    ap.add_argument("--outdir", type=str, default=None,
                    help="Output dir for figures. Default: <run_dir>/export_plots")
    ap.add_argument("--ylog", action="store_true", help="Use log scale on y-axis (values clipped at 1e-12)")

    args = ap.parse_args()

    if (args.logdir is None) == (args.run_dir is None):
        raise SystemExit("Provide exactly one of: --logdir or --run-dir")

    if args.run_dir is not None:
        run_dir = Path(args.run_dir).resolve()
        if not is_run_dir(run_dir):
            raise SystemExit(f"--run-dir must contain tb/: {run_dir}")
    else:
        exp_dir = Path(args.logdir).resolve()
        run_dir = pick_run_dir(exp_dir, args.run_id)

    tb_dir = run_dir / "tb"
    outdir = Path(args.outdir).resolve() if args.outdir else (run_dir / "export_plots")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[run_dir] {run_dir}")
    print(f"[tb_dir]  {tb_dir}")
    print(f"[outdir]  {outdir}")

    event_files = pick_event_files(tb_dir, args.event_policy)

    if args.event_policy == "all" and len(event_files) > 1:
        # save each event into its own subfolder for cleanliness
        for ev in event_files:
            sub = outdir / safe_filename(ev.name)
            sub.mkdir(parents=True, exist_ok=True)
            process_one_event(ev, sub, ylog=args.ylog)
    else:
        process_one_event(event_files[0], outdir, ylog=args.ylog)

    print("Done.")


if __name__ == "__main__":
    main()