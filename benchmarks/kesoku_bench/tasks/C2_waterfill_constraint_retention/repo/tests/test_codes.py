"""Tests for domerr.codes."""

import re

from domerr.codes import COUNTRY_ALIASES, ErrorCode

CODE_RE = re.compile(r"^E_DOM_[A-Z]+_\d{3}$")


def test_codes_follow_naming_scheme() -> None:
    """Every code is E_DOM_<MODULE>_<NNN>."""
    for member in ErrorCode:
        assert CODE_RE.fullmatch(member.value), member


def test_code_values_equal_names() -> None:
    """Values serialise unchanged."""
    for member in ErrorCode:
        assert member.value == member.name


def test_codes_are_unique() -> None:
    """No two members share a value."""
    values = [m.value for m in ErrorCode]
    assert len(values) == len(set(values))


def test_country_aliases_map_to_alpha2() -> None:
    """Alias targets are upper-case two-letter codes."""
    for alias, code in COUNTRY_ALIASES.items():
        assert re.fullmatch(r"[A-Z]{2}", code), alias


def test_country_alias_canonical_codes_map_to_themselves() -> None:
    """Each target code is itself a key mapping to itself."""
    for code in set(COUNTRY_ALIASES.values()):
        assert COUNTRY_ALIASES[code] == code
