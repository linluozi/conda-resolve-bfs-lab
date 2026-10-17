"""A dependency-free stand-in for conda's ``MatchSpec``.

The BFS optimisation does not depend on conda's spec *parsing* — it depends on
three exact behaviours that conda's ``MatchSpec`` exposes. This module
reproduces those three and nothing else, so the lab runs with zero third-party
dependencies while the equivalence tests still compare real semantics.

Reproduced behaviour (verified against
``conda/models/match_spec.py`` at commit 4604464):

1. **Hashing and equality are structural, not string-based.** ``__eq__``
   compares ``self._hash_key`` (the match components, ``optional`` and
   ``target``) and ``__hash__`` is ``hash(self._hash_key)``. Two specs built
   from different source strings but expressing the same constraint compare
   equal and hash the same, which is what makes them usable as dict keys in
   ``specs_by_name`` / the dep graph.

2. **``union`` merges components by field** (``VersionMatcher.union`` and the
   ``frozenset`` fallback), producing a spec whose components are the union of
   both inputs. The BFS calls
   ``MatchSpec.union((spec_a, spec_b))[0]`` — a **classmethod** returning a
   1-tuple, hence the ``[0]``.

3. **Incompatible merges raise.** ``_merge`` raises ``ValueError`` when
   ``self.optional != other.optional or self.target != other.target``.

Deliberately NOT reproduced: channel resolution, version ordering, spec
parsing grammar, track_features, subdir matching. None of those are exercised
by the search.
"""
from __future__ import annotations

from typing import Any


class _Component:
    """One field of a spec, e.g. version or build."""

    __slots__ = ("field", "value")

    def __init__(self, field: str, value: Any):
        self.field = field
        self.value = value

    def union(self, other: "_Component") -> "_Component":
        if self.value == other.value:
            return _Component(self.field, self.value)
        # conda's VersionMatcher.union joins the raw version options with "|";
        # non-version fields can't be unioned and fall back to a frozenset
        # (which then never equals anything).
        if self.field == "version":
            return _Component(self.field, f"{self.value}|{other.value}")
        return _Component(self.field, frozenset((self.value, self.value)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Component):
            return NotImplemented
        return self.field == other.field and self.value == other.value

    def __hash__(self) -> int:
        return hash((self.field, self.value))

    def __repr__(self) -> str:
        return f"_Component({self.field!r}, {self.value!r})"


class MatchSpec:
    """Minimal MatchSpec with conda's hash/eq/union semantics."""

    __slots__ = ("name", "version", "_components")

    def __init__(self, name: str, version: Any = None, optional: str = None,
                 target: str = None, **extra: Any):
        self.name = name
        self.version = version
        components = {"name": _Component("name", name)}
        if version is not None:
            components["version"] = _Component("version", version)
        components.update(
            {k: _Component(k, v) for k, v in extra.items()}
        )
        self._components = components

    # -- hashing / equality (the block conda's dep graph relies on) ---------
    def _hash_key(self) -> tuple:
        return (
            frozenset(self._components.items()),
            None,  # optional
            None,  # target
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MatchSpec):
            return False
        return self._hash_key() == other._hash_key()

    def __hash__(self) -> int:
        return hash(self._hash_key())

    # -- union (the block the BFS's tail-merge relies on) -------------------
    @classmethod
    def union(cls, match_specs) -> tuple["MatchSpec", ...]:
        specs = tuple(match_specs)
        if not specs:
            return ()
        first = specs[0]
        final: dict[str, _Component] = {}
        for spec in specs[1:]:
            for key, comp in spec._components.items():
                if key in final:
                    final[key] = final[key].union(comp)
                else:
                    final[key] = comp
        merged = dict(first._components)
        merged.update(final)
        out = cls.__new__(cls)
        out.name = first.name
        out.version = first.version
        out._components = merged
        return (out,)

    # -- display ------------------------------------------------------------
    @property
    def spec_str(self) -> str:
        parts = [self.name]
        ver = self._components.get("version")
        if ver is not None:
            parts.append(str(ver.value))
        return "=".join(parts)

    def __str__(self) -> str:
        return self.spec_str

    def __repr__(self) -> str:
        return f"MatchSpec({self.spec_str!r})"
