"""Tests for domerr.registry."""

from collections.abc import Iterator

import pytest
from domerr import registry
from domerr.codes import ErrorCode
from domerr.errors import DomainError

TEST_FIELDS = ("zz_test_field", "zz_dup_field")


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Remove test-only validators after each test."""
    yield
    for field in TEST_FIELDS:
        registry.unregister(field)


def test_validator_registers_and_returns_function() -> None:
    """The decorator returns the original function and registers it."""

    def check(value: str) -> str:
        return value.strip()

    decorated = registry.validator("zz_test_field")(check)
    assert decorated is check
    assert registry.get_validator("zz_test_field") is check
    assert "zz_test_field" in registry.registered_fields()


def test_registering_same_function_twice_is_idempotent() -> None:
    """Re-applying the decorator to the same function is harmless."""

    def check(value: str) -> str:
        return value

    registry.validator("zz_test_field")(check)
    registry.validator("zz_test_field")(check)
    assert registry.get_validator("zz_test_field") is check


def test_duplicate_validator_rejected() -> None:
    """A second, different validator for one field is an error."""
    registry.validator("zz_dup_field")(lambda v: v)
    with pytest.raises(DomainError) as info:
        registry.validator("zz_dup_field")(lambda v: v)
    assert info.value.code is ErrorCode.E_DOM_REGISTRY_001
    assert info.value.field_path == "zz_dup_field"


def test_unknown_field_raises_domain_error() -> None:
    """Looking up a field without a validator raises DomainError, not KeyError."""
    with pytest.raises(DomainError) as info:
        registry.get_validator("zz_no_such_field")
    assert info.value.code is ErrorCode.E_DOM_REGISTRY_002
    assert info.value.field_path == "zz_no_such_field"


def test_unregister_removes_validator() -> None:
    """unregister() drops the entry and tolerates unknown names."""
    registry.validator("zz_test_field")(lambda v: v)
    registry.unregister("zz_test_field")
    registry.unregister("zz_test_field")
    assert "zz_test_field" not in registry.registered_fields()


def test_registered_fields_sorted() -> None:
    """Field names are returned sorted."""
    fields = registry.registered_fields()
    assert fields == sorted(fields)
