"""Unit tests for Kesoku Discord Chatbot voice message components."""

import base64
import io
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from kesoku.gateway.chatbot.discord.voice import (
    VoiceFile,
    _generate_pseudo_waveform,
    _get_audio_duration,
    send_voice_message,
)


def test_voice_file_to_dict() -> None:
    """Test that VoiceFile correctly serializes voice message metadata."""
    fp = io.BytesIO(b"dummy audio content")
    voice_file = VoiceFile(
        fp,
        filename="voice.ogg",
        duration_secs=3.14,
        waveform="abcde12345",
    )

    serialized = voice_file.to_dict(0)
    assert serialized["id"] == 0
    assert serialized["filename"] == "voice.ogg"
    assert serialized["duration_secs"] == 3.14
    assert serialized["waveform"] == "abcde12345"


def test_generate_pseudo_waveform() -> None:
    """Test that pseudo waveform generator returns a valid base64 string of 256 bytes."""
    encoded = _generate_pseudo_waveform()
    assert isinstance(encoded, str)
    # Decode it and ensure it is exactly 256 bytes
    decoded = base64.b64decode(encoded.encode("utf-8"))
    assert len(decoded) == 256
    # Verify all values are in the uint8 range (0-255)
    for val in decoded:
        assert 0 <= val <= 255


@pytest.mark.asyncio
async def test_get_audio_duration_success() -> None:
    """Test that _get_audio_duration successfully extracts float duration from ffprobe output."""
    mock_proc = MagicMock()
    mock_proc.stdout = "  45.67 \n"

    with patch("subprocess.run", return_value=mock_proc) as mock_run:
        duration = await _get_audio_duration("/tmp/test.ogg")
        assert duration == 45.67
        mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_get_audio_duration_failure() -> None:
    """Test that _get_audio_duration gracefully returns 0.0 on subprocess error."""
    with patch("subprocess.run", side_effect=Exception("ffprobe failed")) as mock_run:
        duration = await _get_audio_duration("/tmp/test.ogg")
        assert duration == 0.0
        mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_send_voice_message_success() -> None:
    """Test that send_voice_message successfully triggers ffmpeg, gets metadata, and uploads via low-level API."""
    mock_channel = AsyncMock(spec=discord.Thread)
    mock_channel.id = 12345
    mock_channel._state = MagicMock()
    mock_channel._state.http = AsyncMock()
    mock_channel._state.create_message = MagicMock()

    with patch("os.path.exists", return_value=True):
        with patch("os.unlink") as mock_unlink:
            with patch("subprocess.run") as mock_run:
                with patch("kesoku.gateway.chatbot.discord.voice.VoiceFile") as mock_file_class:
                    # Configure mock file instance to serialize properly
                    mock_file_instance = MagicMock(spec=discord.File)
                    mock_file_instance.to_dict.return_value = {"id": 0, "filename": "voice.ogg"}
                    mock_file_class.return_value = mock_file_instance

                    # Configure mock run to return a valid duration for ffprobe call
                    mock_proc = MagicMock()
                    mock_proc.stdout = "3.5\n"
                    mock_run.return_value = mock_proc

                    await send_voice_message(mock_channel, "/tmp/voice.ogg")

                    assert mock_run.call_count == 2
                    mock_unlink.assert_called_once()
                    # Verifies low-level HTTP send_message was called for the voice attachment
                    mock_channel._state.http.send_message.assert_called_once()
