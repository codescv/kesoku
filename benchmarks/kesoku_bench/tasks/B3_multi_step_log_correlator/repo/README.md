# netdiag

Tools for the session recorder:

- `uv run python -m netdiag.diagnose` — live session diagnostic: prints the
  parameters negotiated with the recorder, a dump of 1200 probe packets and
  a loss summary.
- `uv run python -m netdiag.cli stats <dumpfile>` — offline summary of a saved
  packet dump (e.g. `fixtures/capture_01.dump`).
- `uv run python -m netdiag.cli replay <dumpfile>` — dry run of the replay
  client (`netdiag/replay.py`), which re-sends a captured session to the
  recorder using the session parameters in its module constants.

Packet dump format: one packet per line, `<seq:04d> <length:4d>B <status>`
with status `ok`, `drop` or `rtx`; `#` starts a comment (see `netdiag/dump.py`).

Pure standard library, Python >= 3.12.

## Development

```bash
uv run pytest
```
