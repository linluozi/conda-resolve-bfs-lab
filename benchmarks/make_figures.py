"""Generate the report figures from results/ablation.csv.

Three charts, one claim each:

  1. ablation-by-arm.png        which of the three changes actually pays
  2. speedup-vs-nesting.png     the work is bounded by NESTING DEPTH, not nodes
  3. absolute-runtime.png       the same data as wall-clock, to show the scale
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "ablation.csv"
OUT = ROOT / "docs"

ARM_COLORS = {
    "A original": "#b0562a",       # terracotta — conda's own palette family
    "B deque": "#c99a5b",          # amber
    "C deque+set": "#7a8b5c",      # olive
    "D direct-iter": "#3f5d6b",    # slate
}
FAMILY_MARKER = {"shallow": "o", "deep": "s"}


def load():
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    for r in rows:
        for k in ("nodes", "A_ms", "B_ms", "C_ms", "D_ms", "path_len"):
            r[k] = float(r[k])
        for k in ("B_vs_A", "C_vs_A", "D_vs_A", "C_vs_B", "D_vs_C"):
            r[k] = float(r[k])
    return rows


def fig1_ablation_by_arm(rows):
    """Single stacked-bar claim: only ONE of the three changes pays.

    Bars are the incremental gain of each arm over its predecessor, so a
    change that adds nothing is a visible zero-height band rather than being
    hidden inside the cumulative total.
    """
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=160)
    families = sorted({r["family"].split(" n=")[0] for r in rows}, reverse=True)
    x = range(len(families))
    width = 0.5

    for i, fam in enumerate(families):
        sub = [r for r in rows if r["family"].startswith(fam)]
        avg = {
            "A original": 1.0,
            "B deque": sum(r["B_vs_A"] for r in sub) / len(sub),
            "C deque+set": sum(r["C_vs_A"] for r in sub) / len(sub),
            "D direct-iter": sum(r["D_vs_A"] for r in sub) / len(sub),
        }
        order = ["A original", "B deque", "C deque+set", "D direct-iter"]
        # incremental contribution of each arm over the one before it
        deltas = [avg[order[0]]] + [avg[o] / avg[order[k - 1]] for k, o in enumerate(order)][1:]

        bottom = 0.0
        for arm, delta in zip(order, deltas):
            ax.bar(i, delta, width, bottom=bottom, color=ARM_COLORS[arm],
                   edgecolor="white", linewidth=1.2,
                   label=arm if i == 0 else None)
            if delta > 0.15:
                ax.text(i, bottom + delta / 2, f"{delta:.2f}x",
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
            elif delta > 0.02:
                # a near-zero band is too short for an inside label
                ax.annotate(f"{delta:.2f}x", (i, bottom + delta),
                            textcoords="offset points", xytext=(0, -14),
                            ha="center", fontsize=8.5, color="#8a4a22")
            bottom += delta

        ax.annotate(f"total {bottom:.2f}x", (i, bottom),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=9.5, fontweight="bold", color="#333")

    ax.set_xticks(list(x))
    ax.set_xticklabels(["shallow\n(nesting depth 2,\nconda's real limit)",
                        "deep\n(chain, nesting\n= n-1)"], fontsize=9.5)
    ax.set_ylabel("incremental speedup over the previous arm", fontsize=9.5)
    ax.set_title("Which of the three changes actually pays\n"
                 "deque adds ~1.00x; the visited set is where the time goes",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "ablation-by-arm.png", bbox_inches="tight")
    plt.close(fig)


def fig2_speedup_vs_nesting(rows):
    """The headline: at depth 2 the work is bounded by depth, not node count.

    The two families are drawn as two separate panels because they are not one
    curve: plotting them on a shared log axis invites the reader to connect
    points that belong to different graph families.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), dpi=160, sharey=True)

    panels = [
        ("shallow", "shallow nesting — depth 2 (conda's real limit)",
         "speedup does not grow with nodes: work is bounded by depth", "#3f5d6b", "o"),
        ("deep", "deep nesting — chain, depth = n-1",
         "speedup grows with nodes: visited scan is now quadratic", "#b0562a", "s"),
    ]

    for ax, (fam, title, subtitle, color, marker) in zip(axes, panels):
        sub = sorted((r for r in rows if r["family"].startswith(fam)),
                     key=lambda r: r["nodes"])
        ns = [r["nodes"] for r in sub]
        ys = [r["D_vs_A"] for r in sub]
        ax.plot(ns, ys, marker=marker, markersize=8, linewidth=2, color=color)
        for r in sub:
            ax.annotate(f"{r['D_vs_A']:.2f}x", (r["nodes"], r["D_vs_A"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, fontweight="bold")

        lo, hi = min(ys), max(ys)
        # headroom for the value labels sitting ABOVE each marker, not just the
        # markers themselves
        pad = (hi - lo) * 1.1 + 0.35
        ax.set_ylim(0, hi + pad)
        ax.set_xscale("log")
        # one tick per data point, plain numbers: the default log ticks
        # ("3×10^2") add nothing and crowd the axis
        ax.set_xticks(ns)
        ax.set_xticklabels([f"{n:g}" for n in ns], fontsize=9)
        ax.set_xlabel("nodes in the graph (log scale)", fontsize=9.5, labelpad=6)
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=16)
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes,
                fontsize=8.5, color="#555", va="bottom")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("speedup of final version (D)\nvs original (A)", fontsize=9.5)
    fig.suptitle("Nesting depth decides what the optimisation buys you",
                 fontsize=12, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "speedup-vs-nesting.png", bbox_inches="tight")
    plt.close(fig)


def fig3_absolute(rows):
    """Absolute wall-clock, so the reader sees the scale behind the ratios."""
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=160)
    for fam, marker in FAMILY_MARKER.items():
        sub = sorted((r for r in rows if r["family"].startswith(fam)),
                     key=lambda r: r["nodes"])
        ns = [r["nodes"] for r in sub]
        for arm, style in (("A original", "-"), ("D direct-iter", "--")):
            key = "A_ms" if arm == "A original" else "D_ms"
            ax.plot(ns, [r[key] for r in sub], style, marker=marker,
                    markersize=6, linewidth=1.9,
                    color="#b0562a" if arm == "A original" else "#3f5d6b",
                    label=f"{fam} / {arm}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nodes in the graph", fontsize=9.5)
    ax.set_ylabel("wall-clock (ms, log scale)", fontsize=9.5)
    ax.set_title("Absolute time: shallow work is depth-bound, deep work is not\n"
                 "two straight parallel lines = one complexity class apart",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6, which="both")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "absolute-runtime.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    fig1_ablation_by_arm(rows)
    fig2_speedup_vs_nesting(rows)
    fig3_absolute(rows)
    print("wrote:")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.relative_to(ROOT)}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
