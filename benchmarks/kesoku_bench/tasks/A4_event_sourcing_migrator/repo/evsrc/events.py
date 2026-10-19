"""Event envelopes, aggregate state, and schema definitions for evsrc."""

from dataclasses import dataclass, field
from typing import Any

LATEST_VERSION = 3

VERSION_SCHEMA_FIELDS: dict[int, tuple[str, ...]] = {
    1: ("balance",),
    2: ("balance", "currency"),
    3: ("balance", "currency", "tags"),
}


class SequenceGap(ValueError):
    """Raised during strict replay when a sequence number gap is detected."""


class CycleError(ValueError):
    """Raised when registering a schema migrator would introduce a cycle."""


@dataclass(frozen=True)
class EventV1:
    """Version 1 account event payload (`delta` and optional initial `balance`)."""

    aggregate_id: str
    sequence: int
    kind: str
    delta: int = 0


@dataclass(frozen=True)
class EventV2:
    """Version 2 account event payload (adds `currency`)."""

    aggregate_id: str
    sequence: int
    kind: str
    delta: int = 0
    currency: str = "USD"


@dataclass(frozen=True)
class EventV3:
    """Version 3 account event payload (adds `tags`)."""

    aggregate_id: str
    sequence: int
    kind: str
    delta: int = 0
    currency: str = "USD"
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EventEnvelope:
    """Versioned event record stored in the event log or passed to replay."""

    aggregate_id: str
    sequence: int
    version: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event envelope into a plain dictionary."""
        return {
            "aggregate_id": self.aggregate_id,
            "sequence": self.sequence,
            "version": self.version,
            "kind": self.kind,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventEnvelope":
        """Construct an EventEnvelope from a dictionary representation."""
        if "payload" in data and isinstance(data["payload"], dict):
            payload = dict(data["payload"])
        else:
            reserved = {"aggregate_id", "sequence", "version", "kind"}
            payload = {k: v for k, v in data.items() if k not in reserved}
        return cls(
            aggregate_id=str(data["aggregate_id"]),
            sequence=int(data["sequence"]),
            version=int(data.get("version", LATEST_VERSION)),
            kind=str(data.get("kind", "credit")),
            payload=payload,
        )


@dataclass
class Account:
    """Materialized state of an account aggregate folded from events."""

    aggregate_id: str
    balance: int = 0
    currency: str = "USD"
    tags: list[str] = field(default_factory=list)
    last_sequence: int = 0
    applied_sequences: list[int] = field(default_factory=list)

    def to_state_dict(self, version: int = LATEST_VERSION) -> dict[str, Any]:
        """Export the account state fields appropriate for `version`."""
        full = {
            "balance": self.balance,
            "currency": self.currency,
            "tags": list(self.tags),
        }
        allowed = VERSION_SCHEMA_FIELDS.get(version, VERSION_SCHEMA_FIELDS[LATEST_VERSION])
        return {k: full[k] for k in allowed if k in full}
