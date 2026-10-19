"""evsrc: event-sourcing schema migrator, snapshot store, and replay engine."""

from evsrc.events import (
    LATEST_VERSION,
    VERSION_SCHEMA_FIELDS,
    Account,
    CycleError,
    EventEnvelope,
    EventV1,
    EventV2,
    EventV3,
    SequenceGap,
)
from evsrc.registry import (
    MigratorRegistry,
    default_registry,
    downcast_payload,
    register_downcaster,
    register_upcaster,
    upcast_payload,
)
from evsrc.replay import replay, replay_account
from evsrc.store import EventStore, Snapshot
from evsrc.upcasters import (
    downcast_v2_to_v1,
    downcast_v3_to_v2,
    install_default_migrators,
    upcast_v1_to_v2,
    upcast_v2_to_v3,
)

__all__ = [
    "LATEST_VERSION",
    "VERSION_SCHEMA_FIELDS",
    "Account",
    "CycleError",
    "EventEnvelope",
    "EventStore",
    "EventV1",
    "EventV2",
    "EventV3",
    "MigratorRegistry",
    "SequenceGap",
    "Snapshot",
    "default_registry",
    "downcast_payload",
    "downcast_v2_to_v1",
    "downcast_v3_to_v2",
    "install_default_migrators",
    "register_downcaster",
    "register_upcaster",
    "replay",
    "replay_account",
    "upcast_payload",
    "upcast_v1_to_v2",
    "upcast_v2_to_v3",
]
