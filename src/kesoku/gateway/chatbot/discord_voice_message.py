"""Discord voice message processing components for Kesoku AI Agent.

Handles converting, preparing, and sending native Discord voice messages.
"""

import asyncio
import base64
import io
import math
import os
import random
import secrets
import subprocess
import tempfile
from typing import Any

import discord

from kesoku.logger import setup_logger

logger = setup_logger(__name__)


class VoiceFile(discord.File):
    """Subclass of discord.File that includes voice message metadata in its JSON representation."""

    def __init__(
        self,
        fp: str | bytes | os.PathLike[Any] | io.BufferedIOBase,
        filename: str | None = None,
        *,
        description: str | None = None,
        spoiler: bool = False,
        duration_secs: float = 0.0,
        waveform: str | None = None,
    ) -> None:
        """Initialize the VoiceFile with additional voice metadata.

        Args:
            fp: The file or file path to attach.
            filename: The filename to use.
            description: The description of the file.
            spoiler: Whether the file is a spoiler.
            duration_secs: Duration of the audio in seconds.
            waveform: Base64-encoded waveform of the audio.
        """
        super().__init__(fp, filename=filename, description=description, spoiler=spoiler)
        self.duration_secs = duration_secs
        self.waveform = waveform

    def to_dict(self, index: int) -> dict[str, Any]:
        """Convert the attachment to a dictionary representation including voice metadata.

        Args:
            index: The index of the attachment in the message payload.

        Returns:
            Dict representation with metadata.
        """
        payload = super().to_dict(index)
        payload["duration_secs"] = self.duration_secs
        if self.waveform:
            payload["waveform"] = self.waveform
        return payload


def _generate_pseudo_waveform() -> str:
    """Generate a pseudo-random envelope waveform of 256 bytes, base64-encoded.

    Returns:
        Base64 encoded waveform string.
    """
    waveform = []
    for i in range(256):
        # Symmetrical envelope: math.sin(pi * i / 255)
        envelope = math.sin(math.pi * i / 255)
        # Multi-frequency sine oscillations to make it look like real audio
        oscillations = 0.5 * math.sin(2 * math.pi * i / 20) + 0.3 * math.sin(2 * math.pi * i / 8)
        # Slight random noise for realism
        noise = random.uniform(-0.05, 0.05)

        # Combine and scale
        amplitude = envelope * (0.75 + 0.2 * oscillations + noise)
        # Map to uint8 range (0-255)
        val = int(amplitude * 255)
        # Clamp between 4 and 255
        val = max(4, min(255, val))
        waveform.append(val)

    waveform_bytes = bytes(waveform)
    return base64.b64encode(waveform_bytes).decode("utf-8")


async def _get_audio_duration(file_path: str) -> float:
    """Get the duration of an audio file in seconds using ffprobe.

    Args:
        file_path: Path to the audio file.

    Returns:
        Duration of the audio file in seconds, or 0.0 if failed.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
        return float(proc.stdout.strip())
    except Exception as e:
        logger.warning(f"Failed to get audio duration for {file_path}: {e}")
        return 0.0


async def send_voice_message(
    channel: discord.Thread | discord.DMChannel | discord.GroupChannel | discord.TextChannel,
    file_path: str,
) -> discord.Message:
    """Send an audio file to a Discord channel/thread as a native voice message.

    Converts the audio file to OGG Opus format using FFmpeg before uploading.

    Args:
        channel: The Discord channel or thread to send the message to.
        file_path: The absolute local file path of the audio file.

    Returns:
        The sent discord.Message object.
    """
    from discord.http import handle_message_parameters
    from discord.message import MessageFlags

    state = channel._state
    flags = MessageFlags._from_value(0)
    flags.voice = True

    # Create a temporary file for the converted OGG audio
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_ogg:
        temp_ogg_path = temp_ogg.name

    try:
        # Run FFmpeg to convert the input audio file to mono OGG format using the libopus codec at 48kHz
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            file_path,
            "-c:a",
            "libopus",
            "-ac",
            "1",
            "-ar",
            "48000",
            temp_ogg_path,
        ]
        # Run in a separate thread to avoid blocking the asyncio event loop
        await asyncio.to_thread(
            subprocess.run,
            cmd,
            check=True,
            capture_output=True,
        )

        # Retrieve voice message metadata: duration and waveform
        duration = await _get_audio_duration(temp_ogg_path)
        waveform = _generate_pseudo_waveform()

        discord_file = VoiceFile(
            temp_ogg_path,
            filename="voice.ogg",
            duration_secs=duration,
            waveform=waveform,
        )
        nonce = secrets.randbits(64)

        # Prepare parameters with the voice flag and send via low-level HTTP client
        with handle_message_parameters(
            file=discord_file,
            nonce=nonce,
            flags=flags,
        ) as params:
            data = await state.http.send_message(channel.id, params=params)

        return state.create_message(channel=channel, data=data)
    finally:
        # Ensure the temporary file is cleaned up
        def _cleanup():
            if os.path.exists(temp_ogg_path):
                try:
                    os.unlink(temp_ogg_path)
                except Exception as ce:
                    logger.warning(f"Failed to clean up temporary voice file {temp_ogg_path}: {ce}")

        await asyncio.to_thread(_cleanup)
