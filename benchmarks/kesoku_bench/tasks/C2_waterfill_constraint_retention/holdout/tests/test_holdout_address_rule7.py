"""Held-out tests for spec rule 7 (address validators): country aliasing and field_path on every error."""

import pytest
from domerr.errors import DomainError


def _address(**overrides: object) -> dict[str, object]:
    """Return a valid raw address with optional overrides."""
    base: dict[str, object] = {
        "street": "1 Infinite Loop",
        "city": "Cupertino",
        "postal_code": "95014",
        "country": "US",
    }
    base.update(overrides)
    return base


def _validate(value: object) -> object:
    """Import lazily so a missing module fails each test individually."""
    from domerr.validators.address import validate_address

    return validate_address(value)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("United States", "US"),
        ("usa", "US"),
        ("USA", "US"),
        ("deutschland", "DE"),
        ("United Kingdom", "GB"),
    ],
    ids=["united_states", "usa_lower", "usa_upper", "deutschland_lower", "united_kingdom"],
)
def test_country_normalised_through_alias_map(raw: str, expected: str) -> None:
    """Country names/aliases normalise to ISO alpha-2, case-insensitively."""
    result = _validate(_address(country=raw))
    assert result["country"] == expected


def test_country_alpha2_lowercase_accepted() -> None:
    """A lower-case alpha-2 code is normalised to upper case."""
    assert _validate(_address(country="de"))["country"] == "DE"


def test_unknown_country_has_field_path() -> None:
    """Unknown countries raise DomainError at address.country."""
    with pytest.raises(DomainError) as info:
        _validate(_address(country="Atlantis"))
    assert info.value.field_path == "address.country"


def test_missing_postal_code_has_field_path() -> None:
    """A missing postal_code is a DomainError at address.postal_code, never KeyError."""
    raw = _address()
    del raw["postal_code"]
    with pytest.raises(DomainError) as info:
        _validate(raw)
    assert info.value.field_path == "address.postal_code"


@pytest.mark.parametrize("subfield", ["street", "city"])
def test_missing_subfield_has_field_path(subfield: str) -> None:
    """Every missing sub-field reports its own address.<subfield> path."""
    raw = _address()
    del raw[subfield]
    with pytest.raises(DomainError) as info:
        _validate(raw)
    assert info.value.field_path == f"address.{subfield}"


def test_validate_record_normalises_country_and_keeps_field_path() -> None:
    """validate_record() runs the address validator and lets its DomainError through unchanged."""
    import domerr.validators.address  # noqa: F401  (registers the validator even without lazy loading)
    from domerr.validators import validate_record

    assert validate_record({"address": _address(country="United States")})["address"]["country"] == "US"
    raw = _address()
    del raw["postal_code"]
    with pytest.raises(DomainError) as info:
        validate_record({"address": raw})
    assert info.value.field_path == "address.postal_code"
