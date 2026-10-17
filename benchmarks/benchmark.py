"""Interleaved ablation benchmark for the conda Resolve BFS optimisation.

Two graph families, because a single family overstates one axis:

  shallow  nesting depth 2 — conda's real limit
           (``context.unsatisfiable_hints_check_depth``). Work is bounded by the
           NESTING DEPTH, not the node count, which is the honest headline
           result.
  deep     a chain with deep nesting, where full-length paths survive the
           ``sub_graph`` descent and the BFS must visit every node before
           reaching the target. This is the regime where the list-based
           ``visited`` scan actually bites.

Method of measurement, chosen so the numbers cannot drift:

  * arms are round-robin'd inside each round (A,B,C,D,A,B,C,D,...)
  * within a round each arm runs 3 times, best-of-3
  * 7 rounds, report each arm's best across rounds
  * speedups are per-round ratios, never raw cross-run values

Writes ``results/ablation.csv`` and prints the same table.
"""
from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from bfs_arms import ARMS  # noqa: E402
from match_spec_stub import MatchSpec  # noqa: E402

ROUNDS = 7
REPEATS = 3
OUT = _HERE.parent / "results" / "ablation.csv"


# --------------------------------------------------------------------------- #
# graph builders
# --------------------------------------------------------------------------- #
def build_nested(edges, name, depth):
    out: dict = {}
    if depth <= 0:
        return out
    # the chain's last node has no outgoing edge, so use .get
    for adj in edges.get(name, ()):
        out[adj] = build_nested(edges, adj.name, depth - 1)
    return out


def make_shallow(n, fanout=4, depth=2):
    """Conda's real regime: fanout edges, nesting capped at `depth`."""
    names = [f"n{i:05d}" for i in range(n)]
    specs = [MatchSpec(nm) for nm in names]
    by_name = {s.name: s for s in specs}
    # one shared edge set, so every nesting level offers the same children
    edges = {
        names[i]: [by_name[names[j]] for j in range(i + 1, min(i + 1 + fanout, n))]
        for i in range(n)
    }
    graphs = {s: build_nested(edges, s.name, depth) for s in specs}
    # reachable within `depth` hops from node 0
    target = specs[min(n - 1, fanout * depth)]
    return graphs, specs, target


def make_deep_chain(n):
    """Chain edges, nesting deep enough that the full-length path survives."""
    names = [f"n{i:05d}" for i in range(n)]
    specs = [MatchSpec(nm) for nm in names]
    edges = {names[i]: [specs[i + 1]] for i in range(n - 1)}
    graphs = {s: build_nested(edges, s.name, n - 1) for s in specs}
    return graphs, specs, specs[-1]


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def timeit(fn, root, target, dep_graph, repeats=REPEATS):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(root, target.name, dep_graph, 1)
        best = min(best, time.perf_counter() - t0)
    return best


def run_case(label, dep_graph, root, target, rows):
    """Measure all arms interleaved; return the per-arm best times."""
    # reject a case where arms disagree on the returned path length
    lengths = set()
    for _, fn in ARMS:
        res = fn(root, target.name, dep_graph, 1)
        if not res:
            print(f"## {label} -> target {target.name} unreachable, skipped")
            return None
        lengths.add(len(res[0]))
    if len(lengths) != 1:
        raise AssertionError(f"{label}: arms disagree on path length {lengths}")

    path_len = lengths.pop()
    samples = {name: [] for name, _ in ARMS}
    for _ in range(ROUNDS):
        for name, fn in ARMS:
            samples[name].append(timeit(fn, root, target, dep_graph))

    best = {name: min(v) for name, v in samples.items()}
    ratios = {
        name: [a / b for a, b in zip(samples["A original"], samples[name])]
        for name, _ in ARMS
    }

    scale, unit = (1_000_000, "us") if best["A original"] < 0.01 else (1000, "ms")
    print(f"\n## {label}  (nodes={len(dep_graph)}, path_len={path_len})")
    print(f"{'arm':<14}{'best':>12}{'vs A':>9}{'median round ratio':>21}")
    prev = None
    for name, _ in ARMS:
        val = best[name] * scale
        vs_a = best["A original"] / best[name]
        med = statistics.median(ratios[name])
        line = f"{name:<14}{val:>9.1f}{unit:>3}{vs_a:>8.2f}x{med:>17.2f}x"
        if prev:
            line += f"   (step {prev / best[name]:.2f}x)"
        print(line)
        prev = best[name]

    rows.append({
        "family": label,
        "nodes": len(dep_graph),
        "path_len": path_len,
        "A_ms": best["A original"] * 1000,
        "B_ms": best["B deque"] * 1000,
        "C_ms": best["C deque+set"] * 1000,
        "D_ms": best["D direct-iter"] * 1000,
        "B_vs_A": best["A original"] / best["B deque"],
        "C_vs_A": best["A original"] / best["C deque+set"],
        "D_vs_A": best["A original"] / best["D direct-iter"],
        "C_vs_B": best["B deque"] / best["C deque+set"],
        "D_vs_C": best["C deque+set"] / best["D direct-iter"],
    })
    return best


def main():
    print(f"Python {sys.version.split()[0]} on {sys.platform}")
    print(f"{ROUNDS} rounds, arms round-robin'd per round, best-of-{REPEATS} within a round.")

    rows: list[dict] = []

    print("\n--- shallow: nesting depth 2 (conda's real limit) ---")
    for n in (200, 500, 1000):
        g, specs, target = make_shallow(n, 4, 2)
        run_case(f"shallow depth=2 n={n}", g, specs[0], target, rows)

    print("\n--- deep: chain, nesting = n-1 ---")
    for n in (60, 120, 240):
        g, specs, target = make_deep_chain(n)
        run_case(f"deep chain n={n}", g, specs[0], target, rows)

    if rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", newline="\n", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
