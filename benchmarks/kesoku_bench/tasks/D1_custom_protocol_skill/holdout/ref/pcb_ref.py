"""Reference implementation of the PCB1 binary protocol for holdout grading."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any

MAGIC = b"PCB1"
FRAME_TYPE_HEADER = 0x00
FRAME_TYPE_DATA = 0x01
EXPECTED_SEQ_BASE = 4100
EXPECTED_FRAME_COUNT = 24
EXPECTED_SESSION_TAG = "sess-17-prod"


@dataclass(frozen=True)
class PCBFrame:
    """Reference parsed PCB1 frame."""

    frame_index: int
    frame_type: int
    payload: bytes
    crc16: int
    expected_crc16: int
    parity: bytes

    @property
    def crc_valid(self) -> bool:
        """Return True if stored CRC16 equals computed CRC16-CCITT."""
        return self.crc16 == self.expected_crc16


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """Compute CRC16-CCITT (poly=0x1021, init=0xFFFF) over a byte sequence."""
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
    """Encode a non-negative integer as an unsigned LEB128 varint."""
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
    """Decode an unsigned LEB128 varint starting at offset."""
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
    """Compute the parity recovery block for a frame payload."""
    masked = bytearray(len(payload) + 1)
    for idx, b in enumerate(payload):
        mask = 0xA5 ^ ((frame_index * 31 + idx * 17) & 0xFF)
        masked[idx] = b ^ mask
    masked[-1] = (sum(payload) ^ 0x5A) & 0xFF
    return bytes(masked)


def expected_frame_payload(frame_index: int, seq_base: int = EXPECTED_SEQ_BASE) -> bytes:
    """Return the canonical uncorrupted payload for frame_index in session_17.pcb.

    Args:
        frame_index: Frame index in 0..23.
        seq_base: Header sequence base (default 4100).

    Returns:
        Canonical payload bytes for the given frame.
    """
    if frame_index == 0:
        return struct.pack(">IHH", seq_base, 0x0001, EXPECTED_FRAME_COUNT) + EXPECTED_SESSION_TAG.encode("utf-8")
    seq_no = EXPECTED_SEQ_BASE + frame_index
    ts = 1700000000 + frame_index * 10
    reading = 220 + (frame_index * 7) % 40
    status = (frame_index * 103) % 1000
    tag = f"ch={frame_index % 4};sensor=temp_{frame_index:02d}".encode("ascii")
    return struct.pack(">IIhH", seq_no, ts, reading, status) + tag


def serialize_frame(
    frame_index: int,
    frame_type: int,
    payload: bytes,
    *,
    corrupt_byte_offset: int | None = None,
    corrupt_bit_mask: int = 0x08,
) -> bytes:
    """Serialize a frame, optionally flipping a single bit in the stored payload after CRC/parity calculation."""
    hdr = struct.pack(">HB", frame_index, frame_type)
    vlen = encode_varint(len(payload))
    crc = crc16_ccitt(hdr + vlen + payload)
    parity = compute_parity_block(frame_index, payload)
    stored_payload = bytearray(payload)
    if corrupt_byte_offset is not None:
        stored_payload[corrupt_byte_offset] ^= corrupt_bit_mask
    return hdr + vlen + bytes(stored_payload) + struct.pack(">H", crc) + parity


def build_session_17_bytes(*, corrupt_frame_9: bool) -> bytes:
    """Construct either the corrupted or the clean reference bytes for session_17.pcb."""
    chunks: list[bytes] = [MAGIC]
    for idx in range(EXPECTED_FRAME_COUNT):
        ftype = FRAME_TYPE_HEADER if idx == 0 else FRAME_TYPE_DATA
        payload = expected_frame_payload(idx)
        corrupt_offset = 4 if (corrupt_frame_9 and idx == 9) else None
        chunks.append(serialize_frame(idx, ftype, payload, corrupt_byte_offset=corrupt_offset))
    return b"".join(chunks)


def parse_pcb_bytes(data: bytes) -> list[PCBFrame]:
    """Parse raw .pcb bytes into a list of PCBFrame objects."""
    if len(data) < 4 or data[:4] != MAGIC:
        raise ValueError("Invalid .pcb file: missing PCB1 magic header")
    offset = 4
    frames: list[PCBFrame] = []
    while offset < len(data):
        if offset + 3 > len(data):
            raise ValueError(f"Truncated frame header at byte offset {offset}")
        frame_index, frame_type = struct.unpack(">HB", data[offset : offset + 3])
        payload_len, payload_start = decode_varint(data, offset + 3)
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
    """Decode Frame 0 header payload into structured fields."""
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


EXPECTED_FRAME_9_PAYLOAD = expected_frame_payload(9)
EXPECTED_CORRUPTED_BYTES = build_session_17_bytes(corrupt_frame_9=True)
EXPECTED_FIXED_BYTES = build_session_17_bytes(corrupt_frame_9=False)
EXPECTED_ORIGINAL_SHA256 = hashlib.sha256(EXPECTED_CORRUPTED_BYTES).hexdigest()
EXPECTED_FIXED_SHA256 = hashlib.sha256(EXPECTED_FIXED_BYTES).hexdigest()
