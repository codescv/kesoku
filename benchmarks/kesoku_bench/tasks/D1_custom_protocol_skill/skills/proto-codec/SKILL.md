---
name: proto-codec
description: Decode, encode, inspect, and repair custom .pcb binary protocol capture files (PCB1 container format with CRC16-CCITT and frame parity recovery).
version: "1.2.0"
required_permissions:
  - run_shell_command
metadata:
  tags:
    - protocol
    - binary
    - pcb
    - crc
    - codec
    - recorder
---

# Proto-Codec (`.pcb` Binary Container Tooling)

This skill documents the team's custom `.pcb` (`PCB1`) binary telemetry capture container format and the `pcbtool.py` CLI used to inspect, decode, encode, and repair corrupted capture files.

## Binary Container Specification (`PCB1`)

Every `.pcb` file starts with the 4-byte ASCII magic header `PCB1`, followed by a sequence of variable-length frames:

1. **Frame Header (3 bytes + varint)**:
   - `frame_index`: `uint16` big-endian (`0` for the session header frame, `1..N-1` for data frames)
   - `frame_type`: `uint8` (`0x00` = Header Frame, `0x01` = Telemetry Data Frame)
   - `payload_len`: unsigned LEB128 varint encoding the byte length of `payload`
2. **Frame Payload (`payload_len` bytes)**:
   - **Frame 0 (Header Frame, `frame_type = 0x00`)**:
     `uint32` big-endian `seq_base`, `uint16` big-endian `flags`, `uint16` big-endian `frame_count`, followed by the UTF-8 `session_tag`.
   - **Frames 1..N-1 (Data Frames, `frame_type = 0x01`)**:
     `uint32` big-endian `seq_no` (`seq_base + frame_index`), `uint32` big-endian `timestamp_s`, `int16` big-endian `reading_tenths`, `uint16` big-endian `status_code`, followed by ASCII channel/sensor tags.
3. **Frame Integrity Trailer**:
   - `crc16`: `uint16` big-endian **CRC16-CCITT** (`poly=0x1021`, `init=0xFFFF`) computed over `frame_index || frame_type || varint(payload_len) || payload`.
   - `parity`: `payload_len + 1` bytes containing the per-frame parity recovery block (`payload[i] ^ 0xA5 ^ ((frame_index * 31 + i * 17) & 0xFF)` for each byte `i`, followed by a 1-byte checksum `(sum(payload) ^ 0x5A) & 0xFF`). Single-frame bit flips or byte corruptions in `payload` can be losslessly reconstructed from this parity block.

## CLI Usage (`pcbtool.py`)

Run `pcbtool.py` via `uv`:

```bash
# Inspect / decode a .pcb capture and check all frame CRCs
uv run python $SKILL_DIR/scripts/pcbtool.py decode <file> [--json]

# Repair corrupted frames using their per-frame parity blocks
uv run python $SKILL_DIR/scripts/pcbtool.py repair <file> [--out <output_file>] [--recompute-crc] [--keep-header]

# Re-encode a decoded JSON dump back into a .pcb binary container
uv run python $SKILL_DIR/scripts/pcbtool.py encode <json_file> --out <output_file>
```

> [!WARNING]
> **Critical `--keep-header` Requirement on `repair`**:
> By default, running `pcbtool.py repair` **without `--keep-header`** re-normalizes Frame 0 and **resets `seq_base` to `0`** (intended only for anonymizing standalone test captures).
> When repairing a recorded session capture for the recorder, you **MUST** pass `--keep-header` (e.g. `uv run python $SKILL_DIR/scripts/pcbtool.py repair fixtures/session_17.pcb --out fixtures/session_17.fixed.pcb --recompute-crc --keep-header`) so that Frame 0's original `seq_base` is preserved.
