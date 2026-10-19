"""Held-out sanity tests for spec rules 5-6 on the address validator (codes, registration, __all__)."""

import re

import pytest
from domerr import registry
from domerr.codes import ErrorCode
from domerr.errors import DomainError

ADDRESS_CODE_RE = re.compile(r"^E_DOM_ADDRESS_\d{3}$")

BAD_INPUTS: list[object] = [
    {},
    {"street": "1 Main St", "city": "Springfield", "postal_code": "12345"},
    {"street": "1 Main St", "city": "Springfield", "postal_code": "12345", "country": "Atlantis"},
    {"street": "", "city": "Springfield", "postal_code": "12345", "country": "US"},
]


def test_module_exports_validate_address() -> None:
    """The module lists its single public validator in __all__."""
    from domerr.validators import address

    assert "validate_address" in address.__all__
    assert callable(address.validate_address)


def test_registered_under_address() -> None:
    """@validator("address") registers the public function."""
    from domerr.validators import address

    assert registry.get_validator("address") is address.validate_address


@pytest.mark.parametrize("raw", BAD_INPUTS, ids=["empty", "no_country", "unknown_country", "blank_street"])
def test_errors_use_declared_address_codes(raw: object) -> None:
    """Failures raise DomainError with an E_DOM_ADDRESS_<NNN> member of ErrorCode (no KeyError/ValueError)."""
    from domerr.validators.address import validate_address

    with pytest.raises(DomainError) as info:
        validate_address(raw)
    code = ErrorCode(info.value.code)
    assert ADDRESS_CODE_RE.fullmatch(code.value), code


def test_address_codes_declared_in_enum() -> None:
    """At least one E_DOM_ADDRESS code exists in ErrorCode."""
    assert any(ADDRESS_CODE_RE.fullmatch(m.value) for m in ErrorCode)


def test_valid_address_round_trip() -> None:
    """A valid address returns a mapping with the sub-fields stripped."""
    from domerr.validators.address import validate_address

    result = validate_address(
        {"street": " 1 Main St ", "city": "Springfield ", "postal_code": " 12345", "country": "US"}
    )
    assert result["street"] == "1 Main St"
    assert result["city"] == "Springfield"
    assert result["postal_code"] == "12345"
    assert result["country"] == "US"
