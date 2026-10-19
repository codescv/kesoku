"""Directed acyclic graph of named async tasks.

A :class:`DAG` stores nodes in insertion order. Each node has an async callable
and a tuple of dependency names. ``topo_order()`` validates the graph (unknown
dependencies, cycles) and returns a deterministic topological order: among the
nodes that are ready at the same time, insertion order wins.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from .errors import CycleError, DuplicateNodeError, UnknownNodeError

TaskFn = Callable[[], Awaitable[Any]]


@dataclass(frozen=True)
class Node:
    """A single unit of work in the graph.

    Attributes:
        name: Unique node name.
        fn: Zero-argument coroutine function executed by the scheduler.
        deps: Names of the nodes that must succeed before this one runs.
        timeout_s: Optional per-attempt timeout in seconds.
    """

    name: str
    fn: TaskFn
    deps: tuple[str, ...] = ()
    timeout_s: float | None = None


class DAG:
    """A dependency graph of async tasks."""

    def __init__(self) -> None:
        """Create an empty graph."""
        self._nodes: dict[str, Node] = {}

    def add(self, name: str, fn: TaskFn, deps: Iterable[str] = (), *, timeout_s: float | None = None) -> Node:
        """Register a node.

        Dependencies may refer to nodes that are added later; they are checked by
        :meth:`validate` / :meth:`topo_order`.

        Args:
            name: Unique node name.
            fn: Zero-argument coroutine function.
            deps: Names of upstream nodes.
            timeout_s: Optional per-attempt timeout in seconds.

        Returns:
            The created node.

        Raises:
            DuplicateNodeError: If ``name`` is already registered.
        """
        if name in self._nodes:
            raise DuplicateNodeError(f"node {name!r} already exists")
        deps_t = tuple(dict.fromkeys(deps))  # de-duplicate, keep order
        if name in deps_t:
            raise CycleError([name, name])
        node = Node(name=name, fn=fn, deps=deps_t, timeout_s=timeout_s)
        self._nodes[name] = node
        return node

    def __contains__(self, name: object) -> bool:
        """Return True if ``name`` is a node of the graph."""
        return name in self._nodes

    def __len__(self) -> int:
        """Return the number of nodes."""
        return len(self._nodes)

    def __iter__(self) -> Iterator[str]:
        """Iterate over node names in insertion order."""
        return iter(self._nodes)

    def node(self, name: str) -> Node:
        """Return the node called ``name``.

        Raises:
            UnknownNodeError: If there is no such node.
        """
        try:
            return self._nodes[name]
        except KeyError:
            raise UnknownNodeError(name) from None

    def deps(self, name: str) -> tuple[str, ...]:
        """Return the direct upstream nodes of ``name`` (declaration order)."""
        return self.node(name).deps

    def successors(self, name: str) -> list[str]:
        """Return the direct downstream nodes of ``name`` (insertion order)."""
        self.node(name)
        return [n.name for n in self._nodes.values() if name in n.deps]

    def roots(self) -> list[str]:
        """Return the nodes without dependencies."""
        return [n.name for n in self._nodes.values() if not n.deps]

    def leaves(self) -> list[str]:
        """Return the nodes nothing depends on."""
        used = {d for n in self._nodes.values() for d in n.deps}
        return [name for name in self._nodes if name not in used]

    def descendants(self, name: str) -> set[str]:
        """Return every node reachable downstream of ``name``."""
        seen: set[str] = set()
        stack = self.successors(name)
        while stack:
            cur = stack.pop()
            if cur not in seen:
                seen.add(cur)
                stack.extend(self.successors(cur))
        return seen

    def validate(self) -> None:
        """Check that every dependency exists and that the graph is acyclic.

        Raises:
            UnknownNodeError: If a dependency is not a node.
            CycleError: If the graph contains a cycle.
        """
        self.topo_order()

    def topo_order(self) -> list[str]:
        """Return a deterministic topological order (Kahn's algorithm).

        Raises:
            UnknownNodeError: If a dependency is not a node.
            CycleError: If the graph contains a cycle.
        """
        for node in self._nodes.values():
            for dep in node.deps:
                if dep not in self._nodes:
                    raise UnknownNodeError(f"{node.name!r} depends on unknown node {dep!r}")
        indegree = {name: len(node.deps) for name, node in self._nodes.items()}
        ready = [name for name, deg in indegree.items() if deg == 0]
        order: list[str] = []
        while ready:
            cur = ready.pop(0)
            order.append(cur)
            for succ in self.successors(cur):
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    ready.append(succ)
        if len(order) != len(self._nodes):
            raise CycleError(self._find_cycle(set(self._nodes) - set(order)))
        return order

    def _find_cycle(self, candidates: set[str]) -> list[str]:
        """Return one cycle among ``candidates`` (nodes Kahn's algorithm could not order)."""
        # Iterative DFS following dependency edges; a back edge closes the cycle.
        for start in self._nodes:
            if start not in candidates:
                continue
            path: list[str] = [start]
            on_path = {start}
            iters = [iter(self._nodes[start].deps)]
            while iters:
                nxt = next(iters[-1], None)
                if nxt is None:
                    iters.pop()
                    on_path.discard(path.pop())
                    continue
                if nxt not in candidates:
                    continue
                if nxt in on_path:
                    cycle = path[path.index(nxt) :] + [nxt]
                    cycle.reverse()  # report in execution direction (dep -> dependant)
                    return cycle
                path.append(nxt)
                on_path.add(nxt)
                iters.append(iter(self._nodes[nxt].deps))
        return sorted(candidates)  # pragma: no cover - Kahn guarantees a cycle exists
