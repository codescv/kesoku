"""Exception types raised by dagrun."""

from __future__ import annotations


class DagError(Exception):
    """Base class for every error raised by dagrun."""


class DuplicateNodeError(DagError):
    """Raised when a node name is registered twice."""


class UnknownNodeError(DagError, KeyError):
    """Raised when a node (or a dependency) is not part of the graph."""


class CycleError(DagError):
    """Raised when the dependency graph contains a cycle."""

    def __init__(self, cycle: list[str]) -> None:
        """Store the offending cycle.

        Args:
            cycle: Node names forming the cycle; the first name is repeated at the end.
        """
        self.cycle = cycle
        super().__init__("cycle detected: " + " -> ".join(cycle))


class SchedulerStateError(DagError):
    """Raised when a Scheduler is used in an invalid state (e.g. run twice)."""


class UpstreamFailed(DagError):
    """Signal that a node cannot run because one of its upstream nodes did not succeed."""

    def __init__(self, node: str, upstream: str) -> None:
        """Record which node is blocked and by which upstream node.

        Args:
            node: The node that cannot run.
            upstream: The first upstream node (in dependency order) that did not succeed.
        """
        self.node = node
        self.upstream = upstream
        super().__init__(f"{node!r} blocked by upstream {upstream!r}")
