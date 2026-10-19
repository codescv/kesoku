"""Custom exceptions raised by the DataPipe ingestion pipeline."""


class SchemaMismatch(Exception):
    """Raised when schema.json does not match the structure/types of data/*.csv."""


class AuditViolation(Exception):
    """Raised when CSV data violates rules.yaml or row conservation / quarantine conventions."""
