"""Environment variable binding consumer module (uses OldConfig)."""

from __future__ import annotations

from collections.abc import Mapping

from cfgkit.legacy import OldConfig


def from_env(
    environ: Mapping[str, str],
    prefix: str = "CFGKIT_",
) -> OldConfig:
    """Build an ``OldConfig`` from environment variables starting with ``prefix``."""
    cfg = OldConfig()
    if f"{prefix}SERVICE" in environ:
        cfg.set_field("service", environ[f"{prefix}SERVICE"])
    if f"{prefix}VERSION" in environ:
        cfg.set_field("version", environ[f"{prefix}VERSION"])
    if f"{prefix}HOST" in environ:
        cfg.set_field("host", environ[f"{prefix}HOST"])
    if f"{prefix}PORT" in environ:
        cfg.set_field("port", int(environ[f"{prefix}PORT"]))
    if f"{prefix}DEBUG" in environ:
        raw_dbg = environ[f"{prefix}DEBUG"].strip().lower()
        cfg.set_field("debug", raw_dbg in {"1", "true", "yes", "on"})
    if f"{prefix}TIMEOUT_MS" in environ:
        cfg.set_field("timeout_ms", int(environ[f"{prefix}TIMEOUT_MS"]))
    if f"{prefix}TAGS" in environ:
        raw_tags = environ[f"{prefix}TAGS"].strip()
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        cfg.set_field("tags", tags)
    return cfg


def to_env(
    cfg: OldConfig,
    prefix: str = "CFGKIT_",
) -> dict[str, str]:
    """Export ``cfg`` fields to a ``CFGKIT_*`` environment dictionary."""
    return {
        f"{prefix}SERVICE": str(cfg.get_field("service")),
        f"{prefix}VERSION": str(cfg.get_field("version")),
        f"{prefix}HOST": str(cfg.get_field("host")),
        f"{prefix}PORT": str(cfg.get_field("port")),
        f"{prefix}DEBUG": "true" if cfg.get_field("debug") else "false",
        f"{prefix}TIMEOUT_MS": str(cfg.get_field("timeout_ms")),
        f"{prefix}TAGS": ",".join(cfg.get_field("tags") or []),
    }
