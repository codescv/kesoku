"""Address validator."""

from collections.abc import Mapping
from typing import Any

from domerr.codes import COUNTRY_ALIASES, ErrorCode
from domerr.errors import DomainError
from domerr.registry import validator

__all__ = ["validate_address"]

_REQUIRED = ("street", "city", "postal_code", "country")
# Case-insensitive view of the alias table.
_ALIASES = {alias.casefold(): code for alias, code in COUNTRY_ALIASES.items()}


def _required_str(value: Mapping[str, Any], subfield: str) -> str:
    """Return the stripped sub-field or raise a DomainError carrying its path."""
    try:
        raw = value[subfield]
    except KeyError as exc:
        raise DomainError(
            ErrorCode.E_DOM_ADDRESS_002, f"{subfield} is required", field_path=f"address.{subfield}"
        ) from exc
    if not isinstance(raw, str) or not raw.strip():
        raise DomainError(ErrorCode.E_DOM_ADDRESS_002, f"{subfield} is required", field_path=f"address.{subfield}")
    return raw.strip()


@validator("address")
def validate_address(value: Any) -> dict[str, str]:
    """Normalise a postal address; ``country`` becomes ISO-3166 alpha-2 via ``COUNTRY_ALIASES``.

    Args:
        value: Mapping with ``street``, ``city``, ``postal_code`` and ``country``.

    Returns:
        A new dict with stripped sub-fields and an alpha-2 country code.

    Raises:
        DomainError: With ``field_path="address.<subfield>"`` for any invalid sub-field.
    """
    if not isinstance(value, Mapping):
        raise DomainError(ErrorCode.E_DOM_ADDRESS_001, "address must be a mapping", field_path="address")
    result = {name: _required_str(value, name) for name in _REQUIRED}
    code = _ALIASES.get(result["country"].casefold())
    if code is None:
        raise DomainError(
            ErrorCode.E_DOM_ADDRESS_003, f"unknown country {result['country']!r}", field_path="address.country"
        )
    result["country"] = code
    return result
