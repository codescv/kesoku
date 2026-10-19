# DataPipe Nightly Ingestion (`datapipe`)

`datapipe` validates CSV datasets under `data/` against `schema.json` and `rules.yaml` before loading them into the warehouse.

## Running the Nightly Pipeline

```bash
uv run python -m datapipe.pipeline
```

## Team Conventions

Schema inference (`schema.json`) and data fixture auditing (`rules.yaml` and quarantine handling) are documented in the workspace team skills. Do not hand-edit `schema.json` or delete rows from `data/*.csv`.
