"""Validator registry: ``@validator("<field>")`` and lookup by field name."""

from collections.abc import Callable
from typing import Any, TypeVar

from domerr.codes import ErrorCode
from domerr.errors import DomainError

Validator = Callable[[Any], Any]
V = TypeVar("V", bound=Validator)

_VALIDATORS: dict[str, Validator] = {}


def validator(field: str) -> Callable[[V], V]:
    """Register the decorated function as the validator for ``field``.

    Args:
        field: Top-level record field name.

    Returns:
        A decorator that registers the function and returns it unchanged.
    """

    def decorate(func: V) -> V:
        existing = _VALIDATORS.get(field)
        if existing is not None and existing is not func:
            raise DomainError(
                ErrorCode.E_DOM_REGISTRY_001, f"a validator for {field!r} is already registered", field_path=field
            )
        _VALIDATORS[field] = func
        return func

    return decorate


def get_validator(field: str) -> Validator:
    """Return the validator registered for ``field``.

    Validator modules must have been imported for their decorators to run.

    Args:
        field: Top-level record field name.

    Returns:
        The registered validator.

    Raises:
        DomainError: ``E_DOM_REGISTRY_002`` if no validator is registered.
    """
    try:
        return _VALIDATORS[field]
    except KeyError:
        raise DomainError(ErrorCode.E_DOM_REGISTRY_002, f"no validator for {field!r}", field_path=field) from None


def registered_fields() -> list[str]:
    """Return the sorted names of all fields with a registered validator."""
    return sorted(_VALIDATORS)


def unregister(field: str) -> None:
    """Remove the validator for ``field`` if present (used by tests)."""
    _VALIDATORS.pop(field, None)
