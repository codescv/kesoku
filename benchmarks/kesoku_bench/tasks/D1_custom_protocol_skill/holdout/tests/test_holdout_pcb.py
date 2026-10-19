"""Holdout FAIL_TO_PASS tests for D1_custom_protocol_skill."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_pcb_ref() -> ModuleType:
    """Locate and load pcb_ref.py regardless of where the harness placed holdout/ref."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "ref" / "pcb_ref.py",
        here / "pcb_ref.py",
        here.parent / "ref" / "pcb_ref.py",
        here.parent / "holdout" / "ref" / "pcb_ref.py",
        here.parent.parent / "holdout" / "ref" / "pcb_ref.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("pcb_ref", candidate)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["pcb_ref"] = mod
                spec.loader.exec_module(mod)
                return mod
    raise RuntimeError(f"Could not locate pcb_ref.py in any of {candidates}")


pcb_ref = _load_pcb_ref()
REPO_ROOT = Path(__file__).resolve().parent.parent
ORIG_PCB = REPO_ROOT / "fixtures" / "session_17.pcb"
FIXED_PCB = REPO_ROOT / "fixtures" / "session_17.fixed.pcb"


def test_repaired_file_exists_and_original_untouched() -> None:
    """Verify repaired file exists and original session_17.pcb hash is unchanged."""
    assert FIXED_PCB.is_file(), f"Missing repaired file: {FIXED_PCB}"
    assert ORIG_PCB.is_file(), f"Missing original file: {ORIG_PCB}"
    orig_sha = hashlib.sha256(ORIG_PCB.read_bytes()).hexdigest()
    assert orig_sha == pcb_ref.EXPECTED_ORIGINAL_SHA256, (
        f"Original file was modified: expected sha256 {pcb_ref.EXPECTED_ORIGINAL_SHA256}, got {orig_sha}"
    )


def test_repaired_parses_24_frames_with_valid_crcs() -> None:
    """Verify all 24 frames parse cleanly and every frame passes CRC16-CCITT."""
    assert FIXED_PCB.is_file(), f"Missing repaired file: {FIXED_PCB}"
    frames = pcb_ref.parse_pcb_bytes(FIXED_PCB.read_bytes())
    assert len(frames) == pcb_ref.EXPECTED_FRAME_COUNT, f"Expected 24 frames, got {len(frames)}"
    invalid = [f.frame_index for f in frames if not f.crc_valid]
    assert not invalid, f"Frames with invalid CRC16-CCITT: {invalid}"


def test_repaired_preserves_header_seq_base_4100() -> None:
    """Verify Frame 0 header preserves seq_base == 4100 (--keep-header used)."""
    assert FIXED_PCB.is_file(), f"Missing repaired file: {FIXED_PCB}"
    frames = pcb_ref.parse_pcb_bytes(FIXED_PCB.read_bytes())
    assert frames and frames[0].frame_index == 0
    header = pcb_ref.parse_header_payload(frames[0].payload)
    assert header["seq_base"] == pcb_ref.EXPECTED_SEQ_BASE, (
        f"Expected header seq_base == {pcb_ref.EXPECTED_SEQ_BASE}, got {header['seq_base']} "
        "(did you forget --keep-header?)"
    )
    assert header["frame_count"] == pcb_ref.EXPECTED_FRAME_COUNT
    assert header["session_tag"] == pcb_ref.EXPECTED_SESSION_TAG


def test_repaired_frame_9_payload_recovered() -> None:
    """Verify Frame 9 payload was reconstructed from parity rather than just re-CRC'd."""
    assert FIXED_PCB.is_file(), f"Missing repaired file: {FIXED_PCB}"
    frames = pcb_ref.parse_pcb_bytes(FIXED_PCB.read_bytes())
    assert len(frames) == pcb_ref.EXPECTED_FRAME_COUNT
    for idx, frame in enumerate(frames):
        assert frame.frame_index == idx
        expected_payload = pcb_ref.expected_frame_payload(idx)
        assert frame.payload == expected_payload, f"Payload mismatch in frame {idx}"
    assert frames[9].payload == pcb_ref.EXPECTED_FRAME_9_PAYLOAD
