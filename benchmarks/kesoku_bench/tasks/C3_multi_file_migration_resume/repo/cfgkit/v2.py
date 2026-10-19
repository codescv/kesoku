"""Modern immutable ConfigV2 configuration model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

ALPHABETICAL_KEYS: tuple[str, ...] = (
    "debug",
    "host",
    "limits",
    "port",
    "service",
    "tags",
    "timeout_ms",
    "version",
)


class ConfigV2:
    """Modern configuration object with alphabetical ``to_dict()`` serialization."""

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
        """Initialize a ``ConfigV2`` instance."""
        self.service: str = str(service)
        self.version: str = str(version)
        self.host: str = str(host)
        self.port: int = int(port)
        self.debug: bool = bool(debug)
        self.timeout_ms: int = int(timeout_ms)
        self.tags: list[str] = [str(t) for t in (tags or [])]
        self.limits: dict[str, int] = {str(k): int(v) for k, v in (limits or {}).items()}

    def get(self, name: str, default: Any = None) -> Any:
        """Return field ``name`` or ``default`` if unknown."""
        if name in ALPHABETICAL_KEYS:
            return getattr(self, name)
        return default

    def get_field(self, name: str, default: Any = None) -> Any:
        """Compatibility alias for ``get``."""
        return self.get(name, default)

    def with_updates(self, **changes: Any) -> ConfigV2:
        """Return a new ``ConfigV2`` with ``changes`` applied."""
        data = self.to_dict()
        for key, value in changes.items():
            if key not in ALPHABETICAL_KEYS:
                raise KeyError(f"Unknown configuration field: {key}")
            data[key] = value
        return ConfigV2.from_dict(data)

    def replace(self, **changes: Any) -> ConfigV2:
        """Alias for ``with_updates``."""
        return self.with_updates(**changes)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration fields in alphabetical key order."""
        return {
            "debug": self.debug,
            "host": self.host,
            "limits": dict(self.limits),
            "port": self.port,
            "service": self.service,
            "tags": list(self.tags),
            "timeout_ms": self.timeout_ms,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ConfigV2:
        """Construct a ``ConfigV2`` from a dictionary or mapping."""
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

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ConfigV2:
        """Alias for ``from_dict``."""
        return cls.from_dict(data)

    def copy(self) -> ConfigV2:
        """Return a copy of this configuration."""
        return ConfigV2.from_dict(self.to_dict())

    def __eq__(self, other: object) -> bool:
        """Compare configuration equality by field values."""
        if hasattr(other, "to_dict") and callable(other.to_dict):
            return self.to_dict() == other.to_dict()
        return False
