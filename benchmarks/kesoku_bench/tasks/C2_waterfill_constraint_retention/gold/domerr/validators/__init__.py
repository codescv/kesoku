"""Field validators, one module per top-level record field."""

from collections.abc import Mapping
from typing import Any

from domerr.codes import ErrorCode
from domerr.errors import DomainError
from domerr.registry import get_validator

__all__ = ["validate_record"]


def validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every field of ``record`` with its registered validator.

    Args:
        record: Mapping of top-level field name to raw value.

    Returns:
        A new dict of normalised values.

    Raises:
        DomainError: If the record is not a mapping, a field has no validator, or a validator rejects a value.
    """
    if not isinstance(record, Mapping):
        raise DomainError(ErrorCode.E_DOM_RECORD_001, "record must be a mapping")
    return {field: get_validator(field)(value) for field, value in record.items()}
