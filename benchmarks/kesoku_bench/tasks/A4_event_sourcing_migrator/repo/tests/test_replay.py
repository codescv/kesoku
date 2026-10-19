"""Visible unit tests for ordered event replay and Snapshot persistence."""

from evsrc.events import EventEnvelope
from evsrc.replay import replay
from evsrc.store import EventStore, Snapshot


def test_ordered_replay_across_schema_versions() -> None:
    """Verify ordered sequence 1..3 across v1, v2, and v3 events folds into Account."""
    events = [
        EventEnvelope(
            aggregate_id="acc-100",
            sequence=1,
            version=1,
            kind="set_balance",
            payload={"balance": 100},
        ),
        EventEnvelope(
            aggregate_id="acc-100",
            sequence=2,
            version=2,
            kind="credit",
            payload={"delta": 50, "currency": "EUR"},
        ),
        EventEnvelope(
            aggregate_id="acc-100",
            sequence=3,
            version=3,
            kind="debit",
            payload={"delta": 20, "currency": "EUR", "tags": ["priority"]},
        ),
    ]
    account = replay(events)
    assert account.aggregate_id == "acc-100"
    assert account.balance == 130
    assert account.currency == "EUR"
    assert account.tags == ["priority"]
    assert account.last_sequence == 3
    assert account.applied_sequences == [1, 2, 3]


def test_snapshot_upcast_and_resume_replay() -> None:
    """Verify loading a v1 snapshot at v3 and replaying subsequent events."""
    store = EventStore()
    v1_snap = Snapshot.load(
        {
            "aggregate_id": "acc-200",
            "sequence": 2,
            "version": 1,
            "state": {"balance": 500},
        },
        target_version=3,
    )
    store.save_snapshot(v1_snap)
    loaded = store.get_snapshot("acc-200")
    assert loaded is not None
    assert loaded.version == 3
    assert loaded.balance == 500
    assert loaded.currency == "USD"
    assert loaded.tags == []

    account = replay(
        [
            EventEnvelope(
                aggregate_id="acc-200",
                sequence=3,
                version=3,
                kind="credit",
                payload={"delta": 75, "currency": "USD", "tags": ["active"]},
            )
        ],
        initial_snapshot=loaded,
    )
    assert account.balance == 575
    assert account.last_sequence == 3
