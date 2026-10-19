"""Replay engine folding versioned events into an Account aggregate."""

from collections.abc import Iterable
from typing import Any

import evsrc.upcasters as _upcasters  # noqa: F401
from evsrc.events import LATEST_VERSION, Account, EventEnvelope, SequenceGap
from evsrc.registry import MigratorRegistry, default_registry
from evsrc.store import Snapshot


def _coerce_envelope(item: EventEnvelope | dict[str, Any]) -> EventEnvelope:
    """Convert an event dictionary or `EventEnvelope` into an `EventEnvelope`."""
    if isinstance(item, EventEnvelope):
        return item
    return EventEnvelope.from_dict(item)


def _apply_single_event(
    account: Account,
    envelope: EventEnvelope,
    registry: MigratorRegistry,
) -> None:
    """Upcast `envelope` to `LATEST_VERSION` and fold its mutation onto `account`."""
    payload = registry.upcast(envelope.payload, envelope.version, LATEST_VERSION)
    kind = envelope.kind.lower()

    if kind in ("credit", "deposit"):
        account.balance += int(payload.get("delta", payload.get("amount", 0)))
    elif kind in ("debit", "withdraw"):
        account.balance -= int(payload.get("delta", payload.get("amount", 0)))
    elif kind == "set_balance":
        account.balance = int(payload.get("balance", payload.get("delta", 0)))

    if "currency" in payload and payload["currency"]:
        account.currency = str(payload["currency"])

    for tag in payload.get("tags", []):
        if tag not in account.tags:
            account.tags.append(str(tag))

    account.last_sequence = envelope.sequence
    account.applied_sequences.append(envelope.sequence)


def replay(
    events: Iterable[EventEnvelope | dict[str, Any]],
    *,
    aggregate_id: str | None = None,
    initial_snapshot: Snapshot | None = None,
    gaps: str = "strict",
    registry: MigratorRegistry | None = None,
) -> Account:
    """Fold `events` into an `Account` instance.

    Args:
        events: Stream of `EventEnvelope` objects or serialized event dicts.
        aggregate_id: Optional aggregate ID filter; inferred from snapshot or first event if omitted.
        initial_snapshot: Optional starting snapshot state.
        gaps: Gap handling policy (`"strict"` or `"skip"`).
        registry: Optional `MigratorRegistry` (defaults to `default_registry`).
    """
    del gaps
    active_registry = registry if registry is not None else default_registry
    events_list = list(events)
    envelopes = [_coerce_envelope(ev) for ev in events_list]

    target_agg_id = aggregate_id
    if target_agg_id is None:
        if initial_snapshot is not None:
            target_agg_id = initial_snapshot.aggregate_id
        elif envelopes:
            target_agg_id = envelopes[0].aggregate_id
        else:
            target_agg_id = "default"

    if initial_snapshot is not None:
        snap_v3 = Snapshot.load(
            initial_snapshot.to_dict(),
            target_version=LATEST_VERSION,
            registry=active_registry,
        )
        account = Account(
            aggregate_id=target_agg_id,
            balance=int(snap_v3.state.get("balance", 0)),
            currency=str(snap_v3.state.get("currency", "USD")),
            tags=list(snap_v3.state.get("tags", [])),
            last_sequence=snap_v3.sequence,
        )
    else:
        account = Account(aggregate_id=target_agg_id)

    # BUG: Deduplicates by Python object identity rather than `(aggregate_id, sequence)`,
    # and folds events in arrival order without sorting by `sequence` or checking `gaps`.
    seen_object_ids: set[int] = set()
    for raw_item, env in zip(events_list, envelopes, strict=True):
        if id(raw_item) in seen_object_ids:
            continue
        seen_object_ids.add(id(raw_item))
        if env.aggregate_id != target_agg_id:
            continue
        if initial_snapshot is not None and env.sequence <= initial_snapshot.sequence:
            continue
        _apply_single_event(account, env, active_registry)

    return account


replay_account = replay

__all__ = ["SequenceGap", "replay", "replay_account"]
