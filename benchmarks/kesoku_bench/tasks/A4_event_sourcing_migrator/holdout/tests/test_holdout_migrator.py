"""Holdout verification tests for A4_event_sourcing_migrator."""

import pytest
from evsrc.events import EventEnvelope, SequenceGap
from evsrc.registry import MigratorRegistry
from evsrc.replay import replay
from evsrc.store import Snapshot


def test_replay_idempotent_duplicate_keys() -> None:
    """Duplicate `(aggregate_id, sequence)` events (distinct objects) must be ignored."""
    ordered = [
        EventEnvelope("acc-1", 1, 3, "set_balance", {"balance": 100, "currency": "USD", "tags": ["a"]}),
        EventEnvelope("acc-1", 2, 3, "credit", {"delta": 40, "currency": "USD", "tags": ["b"]}),
        EventEnvelope("acc-1", 3, 3, "debit", {"delta": 15, "currency": "USD", "tags": []}),
    ]
    with_duplicates = [
        EventEnvelope("acc-1", 1, 3, "set_balance", {"balance": 100, "currency": "USD", "tags": ["a"]}),
        # Distinct EventEnvelope instance with the same (aggregate_id, sequence) = ("acc-1", 2)
        EventEnvelope("acc-1", 2, 3, "credit", {"delta": 40, "currency": "USD", "tags": ["b"]}),
        EventEnvelope("acc-1", 2, 3, "credit", {"delta": 40, "currency": "USD", "tags": ["b"]}),
        EventEnvelope("acc-1", 3, 3, "debit", {"delta": 15, "currency": "USD", "tags": []}),
        EventEnvelope("acc-1", 1, 3, "set_balance", {"balance": 100, "currency": "USD", "tags": ["a"]}),
    ]

    expected = replay(ordered)
    actual = replay(with_duplicates)

    assert actual.balance == expected.balance == 125
    assert actual.currency == expected.currency == "USD"
    assert actual.tags == expected.tags == ["a", "b"]
    assert actual.last_sequence == 3
    assert actual.applied_sequences == [1, 2, 3]


def test_replay_out_of_order_shuffled_matches_ordered() -> None:
    """Out-of-order events must be sorted by sequence before folding into Account."""
    ordered = [
        EventEnvelope("acc-2", 1, 2, "set_balance", {"balance": 200, "currency": "GBP"}),
        EventEnvelope("acc-2", 2, 3, "credit", {"delta": 50, "currency": "GBP", "tags": ["t1"]}),
        EventEnvelope("acc-2", 3, 1, "set_balance", {"balance": 80}),
        EventEnvelope("acc-2", 4, 3, "credit", {"delta": 20, "currency": "GBP", "tags": ["t2"]}),
    ]
    shuffled = [ordered[3], ordered[1], ordered[0], ordered[2]]

    expected = replay(ordered)
    actual = replay(shuffled)

    assert actual.balance == expected.balance == 100
    assert actual.tags == expected.tags == ["t1", "t2"]
    assert actual.last_sequence == expected.last_sequence == 4
    assert actual.applied_sequences == [1, 2, 3, 4]


def test_replay_gaps_strict_raises_sequence_gap() -> None:
    """`gaps='strict'` must raise SequenceGap when sequence numbers are non-contiguous."""
    gapped_events = [
        EventEnvelope("acc-3", 1, 3, "credit", {"delta": 10}),
        EventEnvelope("acc-3", 3, 3, "credit", {"delta": 20}),
    ]
    with pytest.raises(SequenceGap):
        replay(gapped_events, gaps="strict")

    # Missing initial sequence 1 (starting at 2 with no snapshot) must also raise SequenceGap.
    with pytest.raises(SequenceGap):
        replay([EventEnvelope("acc-3", 2, 3, "credit", {"delta": 10})], gaps="strict")


def test_replay_gaps_skip_continues() -> None:
    """`gaps='skip'` must ignore missing sequence numbers and fold available events in order."""
    gapped_shuffled = [
        EventEnvelope("acc-4", 5, 3, "debit", {"delta": 30, "tags": ["late"]}),
        EventEnvelope("acc-4", 2, 3, "set_balance", {"balance": 100, "tags": ["init"]}),
    ]
    account = replay(gapped_shuffled, gaps="skip")
    assert account.balance == 70
    assert account.tags == ["init", "late"]
    assert account.last_sequence == 5
    assert account.applied_sequences == [2, 5]


def test_snapshot_load_downcast_v3_to_v2_and_v1() -> None:
    """`Snapshot.load(data, target_version=2)` keeps balance/currency and drops tags/unknowns."""
    v3_data = {
        "aggregate_id": "acc-5",
        "sequence": 9,
        "version": 3,
        "state": {
            "balance": 420,
            "currency": "CAD",
            "tags": ["vip", "beta"],
            "extra_v4_field": "should_be_dropped",
        },
    }
    snap_v2 = Snapshot.load(v3_data, target_version=2)
    assert snap_v2.version == 2
    assert snap_v2.balance == 420
    assert snap_v2.currency == "CAD"
    assert "tags" not in snap_v2.state
    assert "extra_v4_field" not in snap_v2.state
    assert set(snap_v2.state.keys()) == {"balance", "currency"}

    snap_v1 = Snapshot.load(v3_data, target_version=1)
    assert snap_v1.version == 1
    assert snap_v1.balance == 420
    assert set(snap_v1.state.keys()) == {"balance"}


def test_registry_rejects_downcaster_cycles() -> None:
    """`register_downcaster` supports multi-step rollback and rejects cycles/invalid edges."""
    reg = MigratorRegistry()
    reg.register_downcaster(3, 2, lambda d: {k: v for k, v in d.items() if k != "tags"})
    reg.register_downcaster(2, 1, lambda d: {k: v for k, v in d.items() if k != "currency"})

    # Self-loop or cycle (1 -> 3 or 2 -> 3 in downcasters) must raise ValueError / CycleError.
    with pytest.raises(ValueError):
        reg.register_downcaster(2, 2, lambda d: d)
    with pytest.raises(ValueError):
        reg.register_downcaster(1, 3, lambda d: d)
    with pytest.raises(ValueError):
        reg.register_downcaster(2, 3, lambda d: d)
