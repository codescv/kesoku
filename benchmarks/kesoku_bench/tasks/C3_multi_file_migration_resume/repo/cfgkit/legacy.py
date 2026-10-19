"""Legacy mutable configuration model with fixed canonical key order."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LEGACY_KEY_ORDER: tuple[str, ...] = (
    "service",
    "version",
    "host",
    "port",
    "debug",
    "timeout_ms",
    "tags",
    "limits",
)


class OldConfig:
    """Legacy mutable configuration object.

    ``to_dict()`` always returns fields in ``LEGACY_KEY_ORDER``:
    ``("service", "version", "host", "port", "debug", "timeout_ms", "tags", "limits")``.
    """

    def __init__(
        self,
        service: str = "app",
        version: str = "1.0",
        host: str = "127.0.0.1",
        port: int = 8080,
        debug: bool = False,
        timeout_ms: int = 5000,
        tags: list[str] | tuple[str, ...] | None = None,
        limits: Mapping[str, int] | None = None,
    ) -> None:
        """Initialize a legacy configuration object."""
        self.service: str = str(service)
        self.version: str = str(version)
        self.host: str = str(host)
        self.port: int = int(port)
        self.debug: bool = bool(debug)
        self.timeout_ms: int = int(timeout_ms)
        self.tags: list[str] = [str(t) for t in (tags or [])]
        self.limits: dict[str, int] = {str(k): int(v) for k, v in (limits or {}).items()}

    def get_field(self, name: str, default: Any = None) -> Any:
        """Return field ``name`` or ``default`` if unknown."""
        if name in LEGACY_KEY_ORDER:
            return getattr(self, name)
        return default

    def set_field(self, name: str, value: Any) -> None:
        """Mutate field ``name`` in place."""
        if name not in LEGACY_KEY_ORDER:
            raise KeyError(f"Unknown configuration field: {name}")
        if name == "port" or name == "timeout_ms":
            setattr(self, name, int(value))
        elif name == "debug":
            setattr(self, name, bool(value))
        elif name == "tags":
            self.tags = [str(t) for t in value]
        elif name == "limits":
            self.limits = {str(k): int(v) for k, v in dict(value).items()}
        else:
            setattr(self, name, str(value))

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration fields in ``LEGACY_KEY_ORDER``."""
        return {
            "service": self.service,
            "version": self.version,
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "timeout_ms": self.timeout_ms,
            "tags": list(self.tags),
            "limits": dict(self.limits),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OldConfig:
        """Construct an ``OldConfig`` from a mapping."""
        return cls(
            service=data.get("service", "app"),
            version=data.get("version", "1.0"),
            host=data.get("host", "127.0.0.1"),
            port=data.get("port", 8080),
            debug=data.get("debug", False),
            timeout_ms=data.get("timeout_ms", 5000),
            tags=data.get("tags"),
            limits=data.get("limits"),
        )

    def copy(self) -> OldConfig:
        """Return a deep copy of this configuration."""
        return OldConfig.from_dict(self.to_dict())

    def __eq__(self, other: object) -> bool:
        """Compare configuration equality by serialized dictionary."""
        if hasattr(other, "to_dict") and callable(other.to_dict):
            return self.to_dict() == other.to_dict()
        return False
