"""Built-in v1..v3 schema migration functions for evsrc."""

from typing import Any

from evsrc.registry import MigratorRegistry, default_registry


def upcast_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate a v1 event or snapshot state dict to v2 by adding default `currency`."""
    out = dict(data)
    out.setdefault("currency", "USD")
    return out


def upcast_v2_to_v3(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate a v2 event or snapshot state dict to v3 by adding default `tags`."""
    out = dict(data)
    if "tags" not in out or out["tags"] is None:
        out["tags"] = []
    else:
        out["tags"] = list(out["tags"])
    return out


def downcast_v3_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Rollback a v3 state dict to v2 by dropping `tags`."""
    out = dict(data)
    out.pop("tags", None)
    return out


def downcast_v2_to_v1(data: dict[str, Any]) -> dict[str, Any]:
    """Rollback a v2 state dict to v1 by dropping `currency`."""
    out = dict(data)
    out.pop("currency", None)
    return out


def install_default_migrators(registry: MigratorRegistry | None = None) -> None:
    """Register the standard v1 <-> v2 <-> v3 upcasters and downcasters on `registry`."""
    target = registry if registry is not None else default_registry
    target.register_upcaster(1, 2, upcast_v1_to_v2)
    target.register_upcaster(2, 3, upcast_v2_to_v3)
    target.register_downcaster(3, 2, downcast_v3_to_v2)
    target.register_downcaster(2, 1, downcast_v2_to_v1)


install_default_migrators(default_registry)
