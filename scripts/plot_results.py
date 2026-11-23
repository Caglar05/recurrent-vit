"""Plot ImageNet-1k validation curves from run directories (README figure).

    python scripts/plot_results.py <out_dir> [--assets assets]

Reads <out_dir>/<run>/metrics.jsonl for the runs listed in RUNS and writes
assets/results-light.png and assets/results-dark.png:
  (a) val top-1 vs. epoch, (b) top-1 when the classifier reads the CLS token after each
  recurrent step, at the last evaluation (bViT paper, Fig. 2a).
Missing runs are skipped. Curves use the raw model weights (not the EMA).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (run directory, label) in fixed colour-slot order.
RUNS = [
    ("vit-s_bvit_s0", "ViT-S"),
    ("deit3-s_deit3_s0", "DeiT III-S"),
    ("bvit-s_bvit_s0", "bViT-S"),
    ("revit-s-e1_deit3_s0", "reViT-S (E=1)"),
]

THEMES = {  # categorical slots 1-4 + chart chrome, per mode
    "light": dict(series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], surface="#fcfcfb",
                  ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7"),
    "dark": dict(series=["#3987e5", "#d95926", "#199e70", "#c98500"], surface="#1a1a19",
                 ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835"),
}


def load(run_dir: Path):
    f = run_dir / "metrics.jsonl"
    if not f.exists():
        return None
    evals = [r for r in map(json.loads, f.open()) if r.get("kind") == "eval"]
    if not evals:
        return None
    epochs = [r["epoch"] for r in evals]
    top1 = [100 * r["val/top1"] for r in evals]
    per_step = [100 * a for a in evals[-1]["val/top1_per_step"]]
    return epochs, top1, per_step


def spread(ys: list[float], gap: float) -> list[float]:
    """Nudge end-label positions apart by at least `gap`, keeping their order."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:], strict=False):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def style(ax, th):
    ax.set_facecolor(th["surface"])
    ax.grid(axis="y", color=th["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(th["axis"])
    ax.tick_params(colors=th["muted"], labelsize=8, length=0, pad=4)


def plot(data, mode: str, path: Path):
    th = THEMES[mode]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9), dpi=200, gridspec_kw={"wspace": 0.42})
    fig.patch.set_facecolor(th["surface"])
    lw = 1.6

    ends = []
    for (label, (ep, top1, _)), color in data:
        a.plot(ep, top1, color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round")
        ends.append((label, ep[-1], top1[-1], color))
    style(a, th)
    a.set_xlim(0, 300)
    a.set_ylim(0, 85)
    a.set_xticks(range(0, 301, 50))
    a.set_xlabel("epoch", color=th["ink2"], fontsize=8.5)
    a.set_ylabel("val top-1 (%)", color=th["ink2"], fontsize=8.5)
    a.set_title("(a) ImageNet-1k validation accuracy", loc="left", color=th["ink"], fontsize=10, pad=10)
    ys = spread([e[2] for e in ends], gap=4.2)
    for (label, x, y, color), y_lab in zip(ends, ys, strict=True):
        a.plot([x], [y], "o", ms=4, color=color, mec=th["surface"], mew=1.2, zorder=3)
        a.annotate(f"{label} {y:.1f}", (x, y), xytext=(306, y_lab), textcoords="data",
                   color=th["ink"], fontsize=7.5, va="center", annotation_clip=False)

    for (label, (_, _, per_step)), color in data:
        steps = range(1, len(per_step) + 1)
        b.plot(steps, per_step, color=color, lw=lw, marker="o", ms=3.5,
               mec=th["surface"], mew=1.0, label=label)
    style(b, th)
    b.set_xticks(range(1, 13))
    b.set_ylim(0, 85)
    b.set_xlabel("recurrent step t (classifier on the CLS token after t blocks)",
                 color=th["ink2"], fontsize=8.5)
    b.set_ylabel("val top-1 (%)", color=th["ink2"], fontsize=8.5)
    b.set_title("(b) Accuracy after each step, final checkpoint", loc="left", color=th["ink"],
                fontsize=10, pad=10)
    leg = b.legend(loc="upper left", fontsize=7.5, frameon=False, labelcolor=th["ink"], handlelength=1.6)
    for h in leg.legend_handles:
        h.set_markeredgecolor(th["surface"])

    fig.savefig(path, facecolor=th["surface"], bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out_dir", type=Path, help="directory containing one sub-directory per run")
    p.add_argument("--assets", type=Path, default=Path("assets"))
    args = p.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    for mode in THEMES:
        data = []
        for (run, label), color in zip(RUNS, THEMES[mode]["series"], strict=True):
            d = load(args.out_dir / run)
            if d is None:
                print(f"skip {run}: no evaluations")
                continue
            data.append(((label, d), color))
        out = args.assets / f"results-{mode}.png"
        plot(data, mode, out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
