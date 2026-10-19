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
    """Fold `events` into an `Account` instance with deduplication, ordering, and gap policy.

    Args:
        events: Stream of `EventEnvelope` objects or serialized event dicts.
        aggregate_id: Optional aggregate ID filter; inferred from snapshot or first event if omitted.
        initial_snapshot: Optional starting snapshot state.
        gaps: Gap handling policy (`"strict"` raises `SequenceGap`, `"skip"` continues).
        registry: Optional `MigratorRegistry` (defaults to `default_registry`).
    """
    if gaps not in ("strict", "skip"):
        raise ValueError(f"Unsupported gaps policy: {gaps!r} (expected 'strict' or 'skip')")

    active_registry = registry if registry is not None else default_registry
    envelopes = [_coerce_envelope(ev) for ev in events]

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

    # Deduplicate by `(aggregate_id, sequence)` preserving first-seen record per key.
    seen_keys: set[tuple[str, int]] = set()
    unique_events: list[EventEnvelope] = []
    for env in envelopes:
        if env.aggregate_id != target_agg_id:
            continue
        if initial_snapshot is not None and env.sequence <= initial_snapshot.sequence:
            continue
        key = (env.aggregate_id, env.sequence)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_events.append(env)

    # Sort out-of-order events by ascending sequence number.
    unique_events.sort(key=lambda e: e.sequence)

    expected_seq = account.last_sequence + 1
    for env in unique_events:
        if env.sequence != expected_seq:
            if gaps == "strict":
                raise SequenceGap(
                    f"Sequence gap for aggregate {target_agg_id!r}: "
                    f"expected {expected_seq}, got {env.sequence}"
                )
        _apply_single_event(account, env, active_registry)
        expected_seq = env.sequence + 1

    return account


replay_account = replay

__all__ = ["SequenceGap", "replay", "replay_account"]
