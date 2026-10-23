"""Plot learning curves for one run group (e.g. runs/baseline_silent/seed*/metrics.jsonl).

    python scripts/plot_training.py --runs runs/baseline_silent --out results/figures/baseline_silent_learning_curves.png

Three panels share the x axis (training games):
  1. learned good team vs the fixed rule-based evil team (good win rate)
  2. learned evil team vs the fixed rule-based good team (evil win rate)
  3. self-play, learned good vs learned evil (good win rate)
Panels 1-2 measure progress against an opponent that never changes, so a flat curve there
means the policy stopped getting stronger. Panel 3 has no fixed target: both teams keep
adapting to each other, so it shows balance, not progress. Curves are moving averages over
~50k games; dotted verticals mark resumed runs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference palette (light mode): surface, text tokens, recessive grid, categorical slots 1-3.
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
RULE_VS_RULE_GOOD = 0.134  # rule good vs rule evil, 1,000 paired evaluation games


def moving_average(x: np.ndarray, y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) < window:
        return x, y
    kernel = np.ones(window) / window
    return x[window - 1:], np.convolve(y, kernel, mode="valid")


def load(seed_dir: Path) -> dict:
    rows = [json.loads(line) for line in (seed_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    per_iter = rows[0]["games"] if "games" in rows[0] else rows[1]["games"] // rows[1]["iteration"]
    evals = np.array([
        (r["games"],
         r["eval"]["policy_good_vs_rule_evil"]["good_win"]["rate"],
         1 - r["eval"]["rule_good_vs_policy_evil"]["good_win"]["rate"])
        for r in rows if "eval" in r
    ])
    selfplay = np.array([(r["games"], r["selfplay"]["good_win"]["rate"]) for r in rows if r.get("selfplay")])
    starts = [json.loads(line)["iteration"] for line in (seed_dir / "run_info.jsonl").read_text(encoding="utf-8").splitlines()]
    resumes = [it * per_iter for it in starts if it > 0]
    return {"evals": evals, "selfplay": selfplay, "resumes": resumes}


def style_axis(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=8)
    ax.set_ylabel(ylabel, color=TEXT2, fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT2, labelsize=8.5, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def reference_line(ax, y: float, label: str, x_max: float) -> None:
    ax.axhline(y, color=TEXT2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(x_max, y + 0.02, label, ha="right", va="bottom", fontsize=8, color=TEXT2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="runs/baseline_silent")
    parser.add_argument("--out", default="results/figures/baseline_silent_learning_curves.png")
    args = parser.parse_args()

    seeds = sorted(p for p in Path(args.runs).glob("seed*") if (p / "metrics.jsonl").exists() and p.name[4:].isdigit())
    data = {int(p.name[4:]): load(p) for p in seeds}
    x_max = max(d["evals"][-1, 0] for d in data.values()) / 1e6

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.2), sharex=True, facecolor=SURFACE)
    style_axis(axes[0], "Learned good team vs rule-based evil team", "good win rate")
    style_axis(axes[1], "Learned evil team vs rule-based good team", "evil win rate")
    style_axis(axes[2], "Self-play: learned good vs learned evil (no fixed target)", "good win rate")
    reference_line(axes[0], RULE_VS_RULE_GOOD, "rule good vs rule evil (0.13)", x_max)
    reference_line(axes[1], 1 - RULE_VS_RULE_GOOD, "rule evil vs rule good (0.87)", x_max)

    resumes = sorted({g for d in data.values() for g in d["resumes"]})
    for ax in axes:
        for g in resumes:
            ax.axvline(g / 1e6, color=GRID, linewidth=1, linestyle=":", zorder=0)
    if resumes:
        axes[0].text(resumes[0] / 1e6, 0.97, " resumed", fontsize=7.5, color=TEXT2, va="top")

    ends: dict[int, list[tuple[int, float, float]]] = {i: [] for i in range(len(axes))}
    for seed, d in data.items():
        color = SERIES[seed % len(SERIES)]
        ev, sp = d["evals"], d["selfplay"]
        panels = [
            moving_average(ev[:, 0], ev[:, 1], 5),  # evals every 20 iterations -> ~50k games
            moving_average(ev[:, 0], ev[:, 2], 5),
            moving_average(sp[:, 0], sp[:, 1], 100),  # one value per iteration -> ~50k games
        ]
        for i, (x, y) in enumerate(panels):
            axes[i].plot(x / 1e6, y, color=color, linewidth=2, label=f"seed {seed}", solid_capstyle="round")
            axes[i].plot(x[-1] / 1e6, y[-1], "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ends[i].append((seed, x[-1] / 1e6, y[-1]))

    # Direct labels in text ink next to each line's end marker, nudged apart so close finishes don't collide.
    min_gap = 0.055
    for i, points in ends.items():
        points.sort(key=lambda p: p[2])
        placed: list[float] = []
        for _, _, y in points:
            placed.append(max(y, placed[-1] + min_gap) if placed else y)
        shift = max(0.0, placed[-1] - 0.98) if placed else 0.0
        for (seed, x, y), y_label in zip(points, placed):
            axes[i].annotate(f"seed {seed}", (x, y), xytext=(x + x_max * 0.012, y_label - shift), textcoords="data",
                             fontsize=8, color=TEXT2, va="center")

    axes[0].legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=TEXT2, bbox_to_anchor=(1.0, 0.18))
    axes[2].set_xlabel("training games (millions)", color=TEXT2, fontsize=9)
    axes[2].set_xlim(0, x_max * 1.09)
    fig.tight_layout(h_pad=2.2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    print(f"wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
