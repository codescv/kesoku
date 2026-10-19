# domerr

Domain error codes and field validators for the customer-record intake service.

## Layout

| Module | Purpose |
|---|---|
| `domerr/codes.py` | `ErrorCode` enum (machine-readable codes) and shared lookup tables |
| `domerr/errors.py` | `DomainError`, the single exception type raised by validators |
| `domerr/registry.py` | `@validator("<field>")` decorator and validator lookup |
| `domerr/validators/` | one module per record field (being built out) |

## Record schema

A customer record is a mapping with these top-level fields:

| Field | Type | Notes |
|---|---|---|
| `email` | `str` | contact address |
| `phone` | `str` | contact number |
| `address` | mapping | required sub-fields: `street`, `city`, `postal_code`, `country` (all strings) |

## Development

```bash
uv sync
uv run pytest
```
