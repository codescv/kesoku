"""Append-only event log store and versioned Snapshot container."""

from dataclasses import dataclass, field
from typing import Any

import evsrc.upcasters as _upcasters  # noqa: F401
from evsrc.events import LATEST_VERSION, Account, EventEnvelope
from evsrc.registry import MigratorRegistry, default_registry


@dataclass
class Snapshot:
    """Versioned point-in-time snapshot of an `Account` aggregate."""

    aggregate_id: str
    sequence: int
    version: int
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def balance(self) -> int:
        """Return the snapshot's `balance` field."""
        return int(self.state.get("balance", 0))

    @property
    def currency(self) -> str | None:
        """Return the snapshot's `currency` field if present."""
        val = self.state.get("currency")
        return str(val) if val is not None else None

    @property
    def tags(self) -> list[str] | None:
        """Return the snapshot's `tags` list if present."""
        val = self.state.get("tags")
        return list(val) if val is not None else None

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to snapshot state or metadata keys."""
        if key in self.state:
            return self.state[key]
        if key in ("aggregate_id", "sequence", "version", "state"):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        """Check whether `key` exists in the snapshot's `state` dictionary."""
        return key in self.state

    def to_dict(self) -> dict[str, Any]:
        """Serialize the snapshot into a dictionary suitable for `Snapshot.load`."""
        return {
            "aggregate_id": self.aggregate_id,
            "sequence": self.sequence,
            "version": self.version,
            "state": dict(self.state),
        }

    @classmethod
    def from_account(cls, account: Account, version: int = LATEST_VERSION) -> "Snapshot":
        """Create a Snapshot from an `Account` at `version`."""
        return cls(
            aggregate_id=account.aggregate_id,
            sequence=account.last_sequence,
            version=version,
            state=account.to_state_dict(version=version),
        )

    @classmethod
    def load(
        cls,
        data: dict[str, Any],
        target_version: int | None = None,
        *,
        registry: MigratorRegistry | None = None,
    ) -> "Snapshot":
        """Load a Snapshot from `data`, migrating to `target_version` if requested."""
        active_registry = registry if registry is not None else default_registry
        src_version = int(data.get("version", LATEST_VERSION))
        dest_version = target_version if target_version is not None else src_version

        if "state" in data and isinstance(data["state"], dict):
            raw_state = dict(data["state"])
        else:
            reserved = {"aggregate_id", "sequence", "version"}
            raw_state = {k: v for k, v in data.items() if k not in reserved}

        if dest_version == src_version:
            migrated_state = raw_state
        elif dest_version > src_version:
            migrated_state = active_registry.upcast(raw_state, src_version, dest_version)
        else:
            raise NotImplementedError(
                f"Snapshot rollback from version {src_version} to {dest_version} is not supported"
            )

        return cls(
            aggregate_id=str(data.get("aggregate_id", "default")),
            sequence=int(data.get("sequence", 0)),
            version=dest_version,
            state=migrated_state,
        )


class EventStore:
    """In-memory append log of `EventEnvelope` records and latest snapshots."""

    def __init__(self, registry: MigratorRegistry | None = None) -> None:
        """Initialize an empty event store."""
        self._registry = registry if registry is not None else default_registry
        self._events: list[EventEnvelope] = []
        self._snapshots: dict[str, Snapshot] = {}

    def append(self, event: EventEnvelope | dict[str, Any]) -> EventEnvelope:
        """Append an event to the store log and return its `EventEnvelope`."""
        env = event if isinstance(event, EventEnvelope) else EventEnvelope.from_dict(event)
        self._events.append(env)
        return env

    def events_for(self, aggregate_id: str) -> list[EventEnvelope]:
        """Return all stored events for `aggregate_id` in arrival order."""
        return [ev for ev in self._events if ev.aggregate_id == aggregate_id]

    def save_snapshot(self, snapshot: Snapshot) -> None:
        """Save `snapshot` as the latest snapshot for `snapshot.aggregate_id`."""
        self._snapshots[snapshot.aggregate_id] = snapshot

    def get_snapshot(
        self,
        aggregate_id: str,
        target_version: int | None = None,
    ) -> Snapshot | None:
        """Retrieve the snapshot for `aggregate_id`, optionally migrated to `target_version`."""
        snap = self._snapshots.get(aggregate_id)
        if snap is None:
            return None
        if target_version is None:
            return snap
        return Snapshot.load(
            snap.to_dict(),
            target_version=target_version,
            registry=self._registry,
        )
