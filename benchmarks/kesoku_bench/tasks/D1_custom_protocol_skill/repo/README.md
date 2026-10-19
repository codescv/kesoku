# PCB Telemetry Recorder (`recorder`)

This repository contains the data models and aggregation helpers for telemetry sessions produced by our hardware capture units.

## Capture Files (`fixtures/`)

Raw binary capture sessions (`*.pcb`) are stored under `fixtures/`.
The `.pcb` container wire format, frame CRC verification, and parity-based repair workflow are documented in the team skills.

## Running Tests

```bash
uv run pytest
```
