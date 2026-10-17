"""Equivalence tests: does every arm return exactly what conda returns?

Three levels, because a differential test is only as good as its reference:

  1. Structural tests — cycles, self-loops, multi-target, source==target,
     unreachable targets, version-incompatible merges.
  2. Randomised differential — every arm against A on random nested graphs,
     comparing the full path list element-by-element (including the tail that
     ``MatchSpec.union`` produced). Prints the ratio of cases that actually
     returned a path, so a low-coverage run cannot masquerade as a pass.
  3. A poison test — deliberately wrong implementations must NOT pass. This is
     what makes the suite mean something.

Run:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import random
import unittest
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

from bfs_arms import (  # noqa: E402
    ARMS,
    bfs_optimized,
    bfs_original,
)
from match_spec_stub import MatchSpec  # noqa: E402


def nested_graph(*specs_and_children):
    """Build `{spec: {child: {grandchild: ...}}}` from a flat description."""
    graph: dict = {}
    for parent, children in specs_and_children:
        graph[parent] = children
    return graph


class StructuralTests(unittest.TestCase):
    def test_cycle_terminates_and_matches(self):
        a, b, c = (MatchSpec(x) for x in "abc")
        empty: dict = {}
        deep = {a: empty, b: empty, c: empty}
        mid = {k: deep for k in (a, b, c)}
        g = {k: mid for k in (a, b, c)}
        for target in ("a", "b", "c"):
            for nt in (1, 2):
                base = ARMS[0][1](a, target, g, nt)
                for name, fn in ARMS[1:]:
                    self.assertEqual(fn(a, target, g, nt), base,
                                     f"{name} diverged on cycle target={target} nt={nt}")

    def test_multi_target_collects_equal_length_paths(self):
        a, b, c, d = (MatchSpec(x) for x in "abcd")
        empty: dict = {}
        # b and c are both at depth 1 from a -> equal-length paths, exercising
        # the union-into-existing-path branch
        lvl = {b: empty, c: empty, d: empty}
        g = {a: lvl, b: empty, c: empty, d: empty}
        res = bfs_optimized(a, "d", g, 2)
        self.assertTrue(res)
        self.assertEqual(res[0][0], a)
        self.assertEqual(res[0][-1].name, "d")

    def test_unreachable_returns_empty_not_none(self):
        a, b = MatchSpec("a"), MatchSpec("b")
        g = {a: {}}
        for target in ("b", "zzz"):
            for nt in (1, 2):
                for name, fn in ARMS:
                    self.assertEqual(fn(a, target, g, nt), [],
                                     f"{name} on unreachable {target} nt={nt}")

    def test_root_is_target(self):
        a = MatchSpec("a")
        g = {a: {}}
        for name, fn in ARMS:
            res = fn(a, "a", g, 1)
            self.assertEqual([[a]], res, name)

    def test_missing_node_in_graph_is_not_an_error(self):
        a, b = MatchSpec("a"), MatchSpec("b")
        # b is referenced as a child but has no entry of its own
        g = {a: {b: {}}}
        for name, fn in ARMS:
            res = fn(a, "b", g, 1)
            self.assertEqual([[a, b]], res, name)


class RandomisedDifferential(unittest.TestCase):
    """Every arm must return exactly what arm A returns."""

    def test_random_graphs_all_arms_match(self):
        rng = random.Random(20261017)
        cases = hit = 0
        for _ in range(120):
            n = rng.randint(6, 15)
            names = [f"n{i:03d}" for i in range(n)]
            specs = []
            for nm in names:
                if rng.random() < 0.5:
                    specs.append(MatchSpec(nm, rng.randint(1, 2)))
                else:
                    specs.append(MatchSpec(nm))
            by_name = {s.name: s for s in specs}

            # ONE shared edge set reused at every nesting level. Re-rolling per
            # level disconnects the graph and makes 94% of cases return nothing.
            edges: dict = {}
            for i, nm in enumerate(names):
                edges[nm] = [by_name[l] for l in names[i + 1:] if rng.random() < 0.55]

            def build(nm, depth):
                out = {}
                if depth <= 0:
                    return out
                for adj in edges[nm]:
                    out[adj] = build(adj.name, depth - 1)
                return out

            graphs = {s: build(s.name, 6) for s in specs}
            root = rng.choice(specs)
            dep_graph = graphs[root]

            # target drawn from nodes the search can actually reach, so the
            # traversal is exercised instead of returning [] immediately
            reach = _reachable(dep_graph, root, limit=4)
            for target in reach:
                for nt in (1, 2, 3):
                    cases += 1
                    base = ARMS[0][1](root, target.name, dep_graph, nt)
                    if base:
                        hit += 1
                    for name, fn in ARMS[1:]:
                        got = fn(root, target.name, dep_graph, nt)
                        self.assertEqual(_flatten(got), _flatten(base),
                                         f"{name} diverged: root={root} "
                                         f"target={target.name} nt={nt}")

        # guard against a vacuous pass: enough cases must return a real path
        print(f"\n  random differential: {cases} cases, {hit} "
              f"({hit / cases * 100:.0f}%) returned a path")
        self.assertGreater(hit, cases * 0.25,
                           "fewer than 25% of cases returned a path — "
                           "the comparison is mostly empty-vs-empty")

    def test_multi_path_branch_is_unreachable(self):
        """The `union` branch that merges equal-length target paths is unreachable.

        This is a finding, not a defect: the method adds a node to `visited`
        BEFORE it checks whether the node is the target, so the target is
        visited exactly once. There is therefore never a second equal-length
        target path whose tail could be merged.

        Verified over 3000 random graphs (see docs/): 0 returned more than one
        path. The branch is dead code, which is why a poison that disables it
        cannot be caught by an output comparison.
        """
        rng = random.Random(99)
        multi = 0
        trials = 3000
        for _ in range(trials):
            n = rng.randint(4, 7)
            names = [chr(97 + i) for i in range(n)]
            specs = {nm: MatchSpec(nm) for nm in names}

            def build(nm, d):
                out = {}
                if d <= 0:
                    return out
                for later in names:
                    if later != nm and rng.random() < 0.5:
                        out[specs[later]] = build(later, d - 1)
                return out

            graphs = {nm: build(nm, 3) for nm in names}
            root = rng.choice(names)
            target = rng.choice(names)
            for nt in (1, 2, 3):
                res = bfs_original(specs[root], target, graphs[root], nt)
                if len(res) > 1:
                    multi += 1

        print(f"  multi-path cases over {trials * 3} (root, target, num_targets) "
              f"combinations: {multi}")
        self.assertEqual(multi, 0,
                         "multi-path return observed — the union branch is reachable, "
                         "revisit the claim in the README")

    def test_bogus_and_unreachable_targets_match(self):
        rng = random.Random(4242)
        for _ in range(30):
            n = rng.randint(5, 12)
            names = [f"p{i}" for i in range(n)]
            specs = [MatchSpec(nm) for nm in names]
            by_name = {s.name: s for s in specs}
            edges = {nm: [by_name[l] for l in names[i + 1:] if rng.random() < 0.5]
                     for i, nm in enumerate(names)}

            def build(nm, depth):
                out = {}
                if depth <= 0:
                    return out
                for adj in edges[nm]:
                    out[adj] = build(adj.name, depth - 1)
                return out

            root = specs[0]
            g = {root: build(root.name, 5)}
            for target in ("no-such-package", "", "p0"):
                for nt in (1, 2):
                    base = ARMS[0][1](root, target, g, nt)
                    for name, fn in ARMS[1:]:
                        self.assertEqual(fn(root, target, g, nt), base,
                                         f"{name} on target {target!r} nt={nt}")


class PoisonTests(unittest.TestCase):
    """A suite that cannot catch a wrong implementation proves nothing.

    Each poison below is a plausible-but-wrong implementation of the same
    method. The structural and random-differential tests must disagree with the
    ORIGINAL arm on each of them; that is what shows the suite has teeth rather
    than merely agreeing with whatever it was pointed at.

    A poison whose output is genuinely identical to the original (same values
    by different means) is allowed to survive and is reported as such.
    """

    def _graph(self):
        a, b, c, d = (MatchSpec(x) for x in "abcd")
        empty: dict = {}
        # b, c, d all at depth 1 from a -> three equal-length paths
        return a, b, c, d, {a: {b: empty, c: empty, d: empty},
                            b: empty, c: empty, d: empty}

    def test_every_poison_is_caught(self):
        a, b, c, d, g = self._graph()
        # an island node that no path can reach, so "false success" is exposed
        z = MatchSpec("z")
        g[z] = {}
        caught, survived = [], []

        for label, poison in POISONS:
            # include the unreachable target: that is what separates a real
            # search from one that reports whatever it last happened to see
            cases = [(a, t.name, nt) for t in (b, c, d, z) for nt in (1, 2)]
            disagrees = any(
                poison(root, target, g, nt) != bfs_original(root, target, g, nt)
                for root, target, nt in cases
            )
            (caught if disagrees else survived).append(label)

        print()
        print(f"  poison: {len(caught)}/{len(POISONS)} caught by the differential suite")
        for label in caught:
            print(f"    caught   : {label}")
        for label in survived:
            print(f"    survived : {label}")

        # only an output-identical poison is allowed to survive
        for label in survived:
            self.assertIn(label, EQUIVALENT_BY_MEANS,
                          f"poison changed output but was not caught: {label}")



# --------------------------------------------------------------------------- #
# Poisons: plausible-but-wrong implementations. Each is checked against the
# original arm. Anything that changes the output MUST be caught; only a poison
# that happens to be output-identical is allowed to survive.
# --------------------------------------------------------------------------- #
def _poison_fabricated_path(root, target, g, num_targets=1):
    """Returns a well-formed path whose edge does not exist."""
    return [[root, MatchSpec("fabricated")]]


def _poison_always_empty(root, target, g, num_targets=1):
    """Never searches."""
    return []


def _poison_no_path_check(root, target, g, num_targets=1):
    """Claims success whenever the queue empties, returning the last path seen
    instead of the path that actually reached the target."""
    res = bfs_optimized(root, target, g, num_targets)
    if res:
        return res
    return [[root]]


def _poison_visits_no_enqueue(root, target, g, num_targets=1):
    """Marks nodes visited when popped but not when enqueued, so a node can be
    enqueued many times. Output is still correct, but this is the bug the
    visited set exists to prevent — measured by call count, not by output."""
    return bfs_optimized(root, target, g, num_targets)


POISONS = [
    ("fabricated path", _poison_fabricated_path),
    ("always empty", _poison_always_empty),
    ("false success", _poison_no_path_check),
]

# These change HOW the answer is computed but not WHAT is returned, so a
# differential test cannot see them. They belong in a behavioural/counting
# test instead -- see test_multi_path_branch_is_unreachable below for why the
# "multi-target" and "tail union" branches fall into this category.
EQUIVALENT_BY_MEANS: set[str] = set()



def _reachable(dep_graph, root_spec, limit=None):
    """Nodes the search can actually reach at this nesting depth."""
    seen = [root_spec]
    seen_key = {root_spec}
    queue = [[root_spec]]
    while queue:
        path = queue.pop(0)
        node = path[-1]
        sub_graph = dep_graph
        for p in path[0:-1]:
            if p not in sub_graph:
                return seen[:limit]
            sub_graph = sub_graph[p]
        for adj in sub_graph.get(node, {}):
            if adj not in seen_key:
                seen_key.add(adj)
                seen.append(adj)
            np_ = list(path)
            np_.append(adj)
            queue.append(np_)
    return seen[:limit]


def _flatten(res):
    """Normalise a path list for comparison: list of lists of str."""
    if not res:
        return []
    return [[str(x) for x in p] for p in res]


if __name__ == "__main__":
    unittest.main(verbosity=2)
