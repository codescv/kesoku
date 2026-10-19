"""Schema migrator registry for upcasting (and downcasting) versioned payloads."""

from collections import deque
from collections.abc import Callable
from typing import Any

from evsrc.events import LATEST_VERSION, CycleError

MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]


def _has_path(
    edges: dict[int, dict[int, MigrationFn]],
    start_v: int,
    target_v: int,
) -> bool:
    """Return True if a directed path exists from `start_v` to `target_v` in `edges`."""
    if start_v == target_v:
        return True
    visited: set[int] = {start_v}
    queue: deque[int] = deque([start_v])
    while queue:
        curr = queue.popleft()
        for nxt in edges.get(curr, {}):
            if nxt == target_v:
                return True
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return False


def _find_migration_chain(
    edges: dict[int, dict[int, MigrationFn]],
    from_v: int,
    to_v: int,
) -> list[MigrationFn]:
    """Find the shortest chain of migration functions from `from_v` to `to_v` via BFS."""
    if from_v == to_v:
        return []
    queue: deque[tuple[int, list[MigrationFn]]] = deque([(from_v, [])])
    visited: set[int] = {from_v}

    while queue:
        curr, path = queue.popleft()
        for nxt, fn in edges.get(curr, {}).items():
            new_path = [*path, fn]
            if nxt == to_v:
                return new_path
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, new_path))

    raise ValueError(f"No migration path registered from version {from_v} to {to_v}")


class MigratorRegistry:
    """Directed graph registry of version-to-version migration functions."""

    def __init__(self) -> None:
        """Initialize empty upcaster and downcaster migration tables."""
        self._upcasters: dict[int, dict[int, MigrationFn]] = {}
        self._downcasters: dict[int, dict[int, MigrationFn]] = {}

    def register_upcaster(self, from_v: int, to_v: int, fn: MigrationFn) -> None:
        """Register an upcaster function from `from_v` to `to_v`, rejecting cycles."""
        if from_v <= 0 or to_v <= 0:
            raise ValueError("Schema versions must be positive integers")
        if from_v == to_v or _has_path(self._upcasters, to_v, from_v):
            raise CycleError(
                f"Registering upcaster {from_v} -> {to_v} would introduce a cycle"
            )
        if from_v > to_v:
            raise ValueError(
                f"Upcaster must move to a higher version, got {from_v} -> {to_v}"
            )
        self._upcasters.setdefault(from_v, {})[to_v] = fn

    def upcast(
        self,
        payload: dict[str, Any],
        from_v: int,
        to_v: int = LATEST_VERSION,
    ) -> dict[str, Any]:
        """Upcast `payload` from `from_v` to `to_v` along registered upcasters."""
        if from_v == to_v:
            return dict(payload)
        if from_v > to_v:
            raise ValueError(f"Cannot upcast from newer version {from_v} to {to_v}")
        chain = _find_migration_chain(self._upcasters, from_v, to_v)
        data = dict(payload)
        for step_fn in chain:
            data = step_fn(dict(data))
        return data


default_registry = MigratorRegistry()


def register_upcaster(
    from_v: int,
    to_v: int,
    fn: MigrationFn,
    *,
    registry: MigratorRegistry | None = None,
) -> None:
    """Register an upcaster on `registry` (defaults to `default_registry`)."""
    target = registry if registry is not None else default_registry
    target.register_upcaster(from_v, to_v, fn)


def upcast_payload(
    payload: dict[str, Any],
    from_v: int,
    to_v: int = LATEST_VERSION,
    *,
    registry: MigratorRegistry | None = None,
) -> dict[str, Any]:
    """Upcast `payload` from `from_v` to `to_v` using `registry` (or `default_registry`)."""
    target = registry if registry is not None else default_registry
    return target.upcast(payload, from_v, to_v)
