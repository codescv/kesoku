"""Tests for domerr.errors."""

import pytest
from domerr.codes import ErrorCode
from domerr.errors import DomainError


def test_domain_error_fields() -> None:
    """Constructor stores code, message and field path."""
    err = DomainError(ErrorCode.E_DOM_CORE_002, "missing", field_path="email")
    assert err.code is ErrorCode.E_DOM_CORE_002
    assert err.message == "missing"
    assert err.field_path == "email"


def test_domain_error_accepts_string_code() -> None:
    """String values are coerced to the enum."""
    err = DomainError("E_DOM_CORE_001")
    assert err.code is ErrorCode.E_DOM_CORE_001
    assert err.field_path is None


def test_domain_error_rejects_undeclared_code() -> None:
    """Undeclared codes are a programming error."""
    with pytest.raises(ValueError):
        DomainError("E_DOM_NOPE_001")


def test_domain_error_str() -> None:
    """String form includes code and path."""
    assert str(DomainError(ErrorCode.E_DOM_CORE_001, "bad", field_path="phone")) == "[E_DOM_CORE_001] phone: bad"
    assert str(DomainError(ErrorCode.E_DOM_CORE_001, "bad")) == "[E_DOM_CORE_001] bad"


def test_domain_error_to_dict() -> None:
    """Serialisation is stable."""
    err = DomainError(ErrorCode.E_DOM_CORE_002, "missing", field_path="address")
    assert err.to_dict() == {"code": "E_DOM_CORE_002", "message": "missing", "field_path": "address"}


def test_domain_error_is_exception() -> None:
    """It can be raised and caught as Exception."""
    with pytest.raises(Exception):
        raise DomainError(ErrorCode.E_DOM_CORE_001)
