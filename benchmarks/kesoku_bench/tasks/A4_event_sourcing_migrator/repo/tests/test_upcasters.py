"""Visible unit tests for schema upcasting and MigratorRegistry."""

import pytest
from evsrc.events import CycleError
from evsrc.registry import MigratorRegistry, upcast_payload


def test_default_upcast_chain_v1_to_v3() -> None:
    """Verify default v1 -> v2 -> v3 upcaster chain populates currency and tags."""
    v1_payload = {"balance": 250}
    v2_payload = upcast_payload(v1_payload, from_v=1, to_v=2)
    assert v2_payload == {"balance": 250, "currency": "USD"}

    v3_payload = upcast_payload(v1_payload, from_v=1, to_v=3)
    assert v3_payload == {"balance": 250, "currency": "USD", "tags": []}


def test_upcaster_registry_rejects_cycles() -> None:
    """Verify registering a self-loop or backward upcaster raises CycleError / ValueError."""
    reg = MigratorRegistry()
    reg.register_upcaster(1, 2, lambda d: d)
    with pytest.raises((CycleError, ValueError)):
        reg.register_upcaster(1, 1, lambda d: d)
    with pytest.raises((CycleError, ValueError)):
        reg.register_upcaster(2, 1, lambda d: d)
