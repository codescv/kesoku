# Migration Contract & Invariants (`MIGRATIONS.md`)

All migration modules in `ledger/migrations/` (`m001` .. `m006`) must satisfy the **Lossless Round-Trip Invariant**:

```python
down(up(state)) == state
```

Specifically:
1. **Pure / Non-Mutating**: `up(state)` and `down(state)` must return a new list of row dictionaries without mutating the caller's input list or row dicts in place.
2. **Exact Restoration**: Applying `up` followed immediately by `down` must reconstruct every original key and value byte-for-byte, including unusual whitespace (multiple consecutive spaces, leading/trailing spaces) and single-token names (mononyms without spaces).
3. **Clean Key Set After `down`**: If `up` records auxiliary metadata (or uses sentinel values) to make a transformation reversible, `down` must remove any auxiliary keys so the resulting dictionary has the exact same keys and values as the input prior to `up`.

## Migration Chain (`m001` – `m006`)

- `m001_init_accounts`: Normalizes `account_id` to `acct-NNN` format and tags `schema_version = 1`.
- `m002_add_currency`: Splits `balance_str` (`"<CCY> <DEC>"`) into `currency` and `amount`.
- `m003_normalize_status`: Replaces boolean `active` with string `status` (`"ACTIVE"` or `"SUSPENDED"`).
- `m004_split_amount_cents`: Converts decimal `amount` into integer `amount_cents`.
- `m005_split_name`: Splits `full_name` into `first_name` and `last_name`.
- `m006_nullable_last_name_audit`: Converts empty `last_name` values to `None` (`NULL`) and populates `last_name_missing` for audit reporting.
