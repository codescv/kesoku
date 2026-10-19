"""CLI utility for decoding, encoding, and repairing .pcb (PCB1) binary capture files."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAGIC = b"PCB1"
FRAME_TYPE_HEADER = 0x00
FRAME_TYPE_DATA = 0x01


@dataclass
class PCBFrame:
    """Represents a single parsed frame from a .pcb container."""

    frame_index: int
    frame_type: int
    payload: bytes
    crc16: int
    expected_crc16: int
    parity: bytes

    @property
    def crc_valid(self) -> bool:
        """Return True if the stored CRC matches the computed CRC16-CCITT."""
        return self.crc16 == self.expected_crc16


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """Compute CRC16-CCITT (poly=0x1021, init=0xFFFF) over a byte sequence.

    Args:
        data: Input bytes to hash.
        init: Initial 16-bit register value.

    Returns:
        16-bit unsigned CRC value.
    """
    crc = init & 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def encode_varint(value: int) -> bytes:
    """Encode a non-negative integer as an unsigned LEB128 varint.

    Args:
        value: Non-negative integer to encode.

    Returns:
        Varint-encoded bytes.
    """
    if value < 0:
        raise ValueError(f"Varint cannot encode negative value: {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return bytes(out)


def decode_varint(buf: bytes, offset: int) -> tuple[int, int]:
    """Decode an unsigned LEB128 varint starting at offset.

    Args:
        buf: Source byte buffer.
        offset: Starting offset in buf.

    Returns:
        Tuple of (decoded_value, next_offset).
    """
    value = 0
    shift = 0
    pos = offset
    while pos < len(buf):
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, pos
        shift += 7
        if shift > 35:
            raise ValueError("Varint too long")
    raise ValueError("Unexpected EOF while decoding varint")


def compute_parity_block(frame_index: int, payload: bytes) -> bytes:
    """Compute the parity recovery block for a frame payload.

    Args:
        frame_index: The 0-based index of the frame.
        payload: Raw uncorrupted payload bytes.

    Returns:
        Parity block of length len(payload) + 1.
    """
    masked = bytearray(len(payload) + 1)
    for idx, b in enumerate(payload):
        mask = 0xA5 ^ ((frame_index * 31 + idx * 17) & 0xFF)
        masked[idx] = b ^ mask
    masked[-1] = (sum(payload) ^ 0x5A) & 0xFF
    return bytes(masked)


def recover_payload_from_parity(frame_index: int, parity: bytes) -> bytes:
    """Recover the original payload bytes from a frame's parity block.

    Args:
        frame_index: The 0-based index of the frame.
        parity: Stored parity block of length payload_len + 1.

    Returns:
        Reconstructed payload bytes.

    Raises:
        ValueError: If the parity block checksum does not validate.
    """
    if len(parity) < 1:
        raise ValueError("Parity block is empty")
    payload_len = len(parity) - 1
    recovered = bytearray(payload_len)
    for idx in range(payload_len):
        mask = 0xA5 ^ ((frame_index * 31 + idx * 17) & 0xFF)
        recovered[idx] = parity[idx] ^ mask
    expected_check = (sum(recovered) ^ 0x5A) & 0xFF
    if expected_check != parity[-1]:
        raise ValueError(
            f"Parity block checksum mismatch in frame {frame_index}: "
            f"expected 0x{expected_check:02x}, got 0x{parity[-1]:02x}"
        )
    return bytes(recovered)


def serialize_frame(frame_index: int, frame_type: int, payload: bytes) -> bytes:
    """Serialize a single frame with header, varint length, payload, CRC16, and parity.

    Args:
        frame_index: 0-based frame index.
        frame_type: Frame type byte (0x00 header, 0x01 data).
        payload: Frame payload bytes.

    Returns:
        Serialized frame bytes.
    """
    hdr = struct.pack(">HB", frame_index, frame_type)
    vlen = encode_varint(len(payload))
    crc_input = hdr + vlen + payload
    crc = crc16_ccitt(crc_input)
    parity = compute_parity_block(frame_index, payload)
    return crc_input + struct.pack(">H", crc) + parity


def parse_pcb_bytes(data: bytes) -> list[PCBFrame]:
    """Parse raw .pcb bytes into a list of PCBFrame objects without raising on CRC errors.

    Args:
        data: Full contents of a .pcb file.

    Returns:
        List of parsed PCBFrame objects.
    """
    if len(data) < 4 or data[:4] != MAGIC:
        raise ValueError("Invalid .pcb file: missing PCB1 magic header")
    offset = 4
    frames: list[PCBFrame] = []
    while offset < len(data):
        if offset + 3 > len(data):
            raise ValueError(f"Truncated frame header at byte offset {offset}")
        frame_index, frame_type = struct.unpack(">HB", data[offset : offset + 3])
        varint_start = offset + 3
        payload_len, payload_start = decode_varint(data, varint_start)
        payload_end = payload_start + payload_len
        crc_end = payload_end + 2
        parity_end = crc_end + payload_len + 1
        if parity_end > len(data):
            raise ValueError(f"Truncated frame {frame_index} at byte offset {offset}")

        payload = data[payload_start:payload_end]
        (stored_crc,) = struct.unpack(">H", data[payload_end:crc_end])
        parity = data[crc_end:parity_end]
        expected_crc = crc16_ccitt(data[offset:payload_end])

        frames.append(
            PCBFrame(
                frame_index=frame_index,
                frame_type=frame_type,
                payload=payload,
                crc16=stored_crc,
                expected_crc16=expected_crc,
                parity=parity,
            )
        )
        offset = parity_end
    return frames


def parse_header_payload(payload: bytes) -> dict[str, Any]:
    """Decode Frame 0 header payload into structured fields.

    Args:
        payload: Raw payload of Frame 0.

    Returns:
        Dictionary with seq_base, flags, frame_count, and session_tag.
    """
    if len(payload) < 8:
        raise ValueError("Header payload too short")
    seq_base, flags, frame_count = struct.unpack(">IHH", payload[:8])
    session_tag = payload[8:].decode("utf-8", errors="replace")
    return {
        "seq_base": seq_base,
        "flags": flags,
        "frame_count": frame_count,
        "session_tag": session_tag,
    }


def build_header_payload(seq_base: int, flags: int, frame_count: int, session_tag: str) -> bytes:
    """Encode Frame 0 header fields into payload bytes.

    Args:
        seq_base: Base sequence number for the session.
        flags: 16-bit session flags.
        frame_count: Total number of frames in the file.
        session_tag: Session identifier string.

    Returns:
        Packed Frame 0 payload bytes.
    """
    return struct.pack(">IHH", seq_base, flags, frame_count) + session_tag.encode("utf-8")


def parse_data_payload(payload: bytes) -> dict[str, Any]:
    """Decode a telemetry data frame payload into structured fields.

    Args:
        payload: Raw payload of a data frame.

    Returns:
        Dictionary with seq_no, timestamp_s, reading_tenths, status_code, and meta.
    """
    if len(payload) < 12:
        return {"raw_hex": payload.hex()}
    seq_no, timestamp_s, reading_tenths, status_code = struct.unpack(">IIhH", payload[:12])
    meta = payload[12:].decode("ascii", errors="replace")
    return {
        "seq_no": seq_no,
        "timestamp_s": timestamp_s,
        "reading_tenths": reading_tenths,
        "status_code": status_code,
        "meta": meta,
    }


def cmd_decode(args: argparse.Namespace) -> int:
    """Handle the `decode` subcommand."""
    path = Path(args.file)
    frames = parse_pcb_bytes(path.read_bytes())
    corrupted: list[int] = []
    frame_records: list[dict[str, Any]] = []

    for frame in frames:
        rec: dict[str, Any] = {
            "frame_index": frame.frame_index,
            "frame_type": "header" if frame.frame_type == FRAME_TYPE_HEADER else "data",
            "payload_len": len(frame.payload),
            "crc16": f"0x{frame.crc16:04x}",
            "expected_crc16": f"0x{frame.expected_crc16:04x}",
            "crc_valid": frame.crc_valid,
            "payload_hex": frame.payload.hex(),
        }
        if frame.frame_type == FRAME_TYPE_HEADER:
            rec["header"] = parse_header_payload(frame.payload)
        else:
            rec["data"] = parse_data_payload(frame.payload)
        if not frame.crc_valid:
            corrupted.append(frame.frame_index)
        frame_records.append(rec)

    if args.json:
        print(
            json.dumps(
                {
                    "file": str(path),
                    "frame_count": len(frames),
                    "corrupted_frames": corrupted,
                    "frames": frame_records,
                },
                indent=2,
            )
        )
    else:
        hdr_info = frame_records[0].get("header", {}) if frame_records else {}
        print(
            f"File: {path} | Frames: {len(frames)} | "
            f"seq_base={hdr_info.get('seq_base')} | tag={hdr_info.get('session_tag')}"
        )
        for rec in frame_records:
            status = "OK" if rec["crc_valid"] else "CRC_ERROR"
            print(
                f"  Frame {rec['frame_index']:02d} ({rec['frame_type']}): "
                f"len={rec['payload_len']} crc={rec['crc16']} expected={rec['expected_crc16']} [{status}]"
            )
        if corrupted:
            print(f"ERROR: CRC mismatch detected in frame(s): {corrupted}", file=sys.stderr)

    return 1 if corrupted else 0


def cmd_repair(args: argparse.Namespace) -> int:
    """Handle the `repair` subcommand."""
    in_path = Path(args.file)
    frames = parse_pcb_bytes(in_path.read_bytes())

    if args.out:
        out_path = Path(args.out)
    elif args.output_pos:
        out_path = Path(args.output_pos)
    elif in_path.name.endswith(".fixed.pcb"):
        out_path = in_path
    elif in_path.suffix == ".pcb":
        out_path = in_path.with_name(f"{in_path.stem}.fixed.pcb")
    else:
        out_path = Path(f"{in_path}.fixed")

    repaired_indices: list[int] = []
    out_chunks: list[bytes] = [MAGIC]

    for frame in frames:
        payload = frame.payload
        if not frame.crc_valid:
            payload = recover_payload_from_parity(frame.frame_index, frame.parity)
            repaired_indices.append(frame.frame_index)

        if frame.frame_index == 0 and frame.frame_type == FRAME_TYPE_HEADER and not args.keep_header:
            hdr = parse_header_payload(payload)
            # Without --keep-header, seq_base is reset to 0 as documented in SKILL.md.
            payload = build_header_payload(
                seq_base=0,
                flags=hdr["flags"],
                frame_count=hdr["frame_count"],
                session_tag=hdr["session_tag"],
            )

        out_chunks.append(serialize_frame(frame.frame_index, frame.frame_type, payload))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"".join(out_chunks))

    print(
        f"Repaired {len(repaired_indices)} corrupted frame(s) {repaired_indices} -> {out_path} "
        f"(keep_header={bool(args.keep_header)})"
    )
    return 0


def cmd_encode(args: argparse.Namespace) -> int:
    """Handle the `encode` subcommand from a JSON specification."""
    in_path = Path(args.file)
    doc = json.loads(in_path.read_text(encoding="utf-8"))
    out_path = Path(args.out) if args.out else in_path.with_suffix(".pcb")

    out_chunks: list[bytes] = [MAGIC]
    for item in doc.get("frames", []):
        f_idx = int(item["frame_index"])
        f_type = FRAME_TYPE_HEADER if item.get("frame_type") in (0, "header") else FRAME_TYPE_DATA
        payload = bytes.fromhex(item["payload_hex"])
        out_chunks.append(serialize_frame(f_idx, f_type, payload))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"".join(out_chunks))
    print(f"Encoded {len(out_chunks) - 1} frames -> {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for pcbtool CLI."""
    parser = argparse.ArgumentParser(description="PCB1 binary protocol capture tool")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_dec = sub.add_parser("decode", help="Decode and verify a .pcb capture file")
    p_dec.add_argument("file", help="Path to .pcb file")
    p_dec.add_argument("--json", action="store_true", help="Output decoded structure as JSON")

    p_rep = sub.add_parser("repair", help="Repair corrupted frames using parity recovery blocks")
    p_rep.add_argument("file", help="Path to input .pcb file")
    p_rep.add_argument("output_pos", nargs="?", default=None, help="Optional output .pcb file path")
    p_rep.add_argument("-o", "--out", "--output", dest="out", default=None, help="Output .pcb file path")
    p_rep.add_argument(
        "--recompute-crc",
        action="store_true",
        help="Recompute frame CRC16-CCITT after parity recovery",
    )
    p_rep.add_argument(
        "--keep-header",
        action="store_true",
        help="Preserve original Frame 0 header (including seq_base) instead of resetting seq_base to 0",
    )

    p_enc = sub.add_parser("encode", help="Encode a JSON frame dump into a .pcb file")
    p_enc.add_argument("file", help="Path to JSON file")
    p_enc.add_argument("-o", "--out", "--output", dest="out", required=True, help="Output .pcb file path")

    args = parser.parse_args(argv)
    if args.subcommand == "decode":
        return cmd_decode(args)
    if args.subcommand == "repair":
        return cmd_repair(args)
    if args.subcommand == "encode":
        return cmd_encode(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
