"""conda Resolve BFS — four-arm reconstruction.

Each function is one step of a three-part optimization applied to
``conda/resolve.py: Resolve.breadth_first_search_for_dep_graph`` (commit 4604464,
2026-10-16). Each arm adds exactly one change to the previous one, so every
speedup ratio in the report can be attributed to a single change.

Three behaviours prevent a parent-map rewrite (the usual "textbook" fix):

1. The child set depends on the path taken to reach the node:

       sub_graph = dep_graph
       for p in path[0:-1]:
           sub_graph = sub_graph[p]

   so the same node reached by different paths can have different children.
   A single global parent/visited map is therefore NOT semantically safe here.

2. The method collects up to ``num_targets`` EQUAL-LENGTH paths, not just the
   first one, and merges a matching-length path's tail into the stored one:

       if len(target_paths[-1]) == len(path):
           last_spec = MatchSpec.union((path[-1], target_paths[-1][-1]))[0]
           target_paths[-1][-1] = last_spec
       else:
           target_paths.append(path)

3. The original contains dead code:

       children = [_ for _ in sub_graph.get(node, {})]
       if children is None:
           continue

   ``dict.get(key, {})`` never returns ``None``, so the guard can never fire.

The three changes that ARE safe, and are what this repo measures:

  A  original     list queue + pop(0), list ``visited``, materialised children
  B  deque        A with ``collections.deque.popleft()``
  C  deque+set    B with a set for ``visited`` membership
  D  direct-iter  C iterating the dependency mapping directly (final version)
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

# This repo is a standalone lab; it does not require conda. The MatchSpec
# import is only needed for the union call in arm A/B/C/D and is satisfied by
# the repo's lightweight stand-in below. See tests/test_arms.py for how
# equivalence is checked without installing conda.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from match_spec_stub import MatchSpec  # noqa: E402


def bfs_original(root_spec, target_name, dep_graph, num_targets=1):
    """Verbatim body of conda's current method."""
    queue = []
    queue.append([root_spec])
    visited = []
    target_paths = []
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node in visited:
            continue
        visited.append(node)
        if node.name == target_name:
            if len(target_paths) == 0:
                target_paths.append(path)
            if len(target_paths[-1]) == len(path):
                last_spec = MatchSpec.union((path[-1], target_paths[-1][-1]))[0]
                target_paths[-1][-1] = last_spec
            else:
                target_paths.append(path)

            found_all_targets = len(target_paths) == num_targets and any(
                len(_) != len(path) for _ in queue
            )
            if len(queue) == 0 or found_all_targets:
                return target_paths
        sub_graph = dep_graph
        for p in path[0:-1]:
            sub_graph = sub_graph[p]
        children = [_ for _ in sub_graph.get(node, {})]
        if children is None:
            continue
        for adj in children:
            if len(target_paths) < num_targets:
                new_path = list(path)
                new_path.append(adj)
                queue.append(new_path)
    return target_paths


def bfs_deque(root_spec, target_name, dep_graph, num_targets=1):
    """A + deque queue. Nothing else changed."""
    queue = deque([[root_spec]])
    visited = []
    target_paths = []
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node in visited:
            continue
        visited.append(node)
        if node.name == target_name:
            if len(target_paths) == 0:
                target_paths.append(path)
            if len(target_paths[-1]) == len(path):
                last_spec = MatchSpec.union((path[-1], target_paths[-1][-1]))[0]
                target_paths[-1][-1] = last_spec
            else:
                target_paths.append(path)

            found_all_targets = len(target_paths) == num_targets and any(
                len(_) != len(path) for _ in queue
            )
            if len(queue) == 0 or found_all_targets:
                return target_paths
        sub_graph = dep_graph
        for p in path[0:-1]:
            sub_graph = sub_graph[p]
        children = [_ for _ in sub_graph.get(node, {})]
        if children is None:
            continue
        for adj in children:
            if len(target_paths) < num_targets:
                new_path = list(path)
                new_path.append(adj)
                queue.append(new_path)
    return target_paths


def bfs_deque_set(root_spec, target_name, dep_graph, num_targets=1):
    """B + set-based visited membership. Still materialises the children list."""
    queue = deque([[root_spec]])
    visited = set()
    target_paths = []
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node in visited:
            continue
        visited.add(node)
        if node.name == target_name:
            if len(target_paths) == 0:
                target_paths.append(path)
            if len(target_paths[-1]) == len(path):
                last_spec = MatchSpec.union((path[-1], target_paths[-1][-1]))[0]
                target_paths[-1][-1] = last_spec
            else:
                target_paths.append(path)

            found_all_targets = len(target_paths) == num_targets and any(
                len(_) != len(path) for _ in queue
            )
            if len(queue) == 0 or found_all_targets:
                return target_paths
        sub_graph = dep_graph
        for p in path[0:-1]:
            sub_graph = sub_graph[p]
        children = [_ for _ in sub_graph.get(node, {})]
        if children is None:
            continue
        for adj in children:
            if len(target_paths) < num_targets:
                new_path = list(path)
                new_path.append(adj)
                queue.append(new_path)
    return target_paths


def bfs_optimized(root_spec, target_name, dep_graph, num_targets=1):
    """C + direct iteration over the dependency mapping (final version).

    ``sub_graph.get(node) or ()`` also folds the dead ``if children is None``
    guard, which can never fire because ``dict.get(key, {})`` returns ``{}``.
    """
    queue = deque([[root_spec]])
    visited = set()
    target_paths = []
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node in visited:
            continue
        visited.add(node)
        if node.name == target_name:
            if len(target_paths) == 0:
                target_paths.append(path)
            if len(target_paths[-1]) == len(path):
                last_spec = MatchSpec.union((path[-1], target_paths[-1][-1]))[0]
                target_paths[-1][-1] = last_spec
            else:
                target_paths.append(path)

            found_all_targets = len(target_paths) == num_targets and any(
                len(_) != len(path) for _ in queue
            )
            if len(queue) == 0 or found_all_targets:
                return target_paths
        sub_graph = dep_graph
        for p in path[0:-1]:
            sub_graph = sub_graph[p]
        for adj in sub_graph.get(node) or ():
            if len(target_paths) < num_targets:
                new_path = list(path)
                new_path.append(adj)
                queue.append(new_path)
    return target_paths


ARMS = [
    ("A original", bfs_original),
    ("B deque", bfs_deque),
    ("C deque+set", bfs_deque_set),
    ("D direct-iter", bfs_optimized),
]
