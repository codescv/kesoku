"""Pure-Python WebSocket server implementing the Open-XiaoAI speaker wire protocol."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from kesoku.logger import setup_logger

logger = setup_logger(__name__)

AudioCallback = Callable[[np.ndarray], Awaitable[None]]
EventCallback = Callable[[str, Any], Awaitable[None]]
LifecycleCallback = Callable[[], Awaitable[None]]

PLAY_SAMPLE_RATE = 24000
RECORD_SAMPLE_RATE = 16000
PLAY_CHUNK_BYTES = 4800  # 100ms at 24kHz 16-bit mono


def decode_a113_s32_to_s16(raw_bytes: bytes) -> np.ndarray:
    """Convert Amlogic A113 S32_LE PDM samples (lower 24 bits) to int16 PCM.

    Args:
        raw_bytes: Raw little-endian 32-bit signed integer PCM bytes from LX06 `arecord -f S32_LE`.

    Returns:
        1-D numpy int16 array of 16kHz mono PCM samples.
    """
    n_samples = len(raw_bytes) // 4
    if n_samples == 0:
        return np.empty(0, dtype=np.int16)
    s32 = np.frombuffer(raw_bytes[: n_samples * 4], dtype="<i4")
    shifted = np.right_shift(s32, 8)
    return np.clip(shifted, -32768, 32767).astype(np.int16)


def s16_to_float32(samples: np.ndarray) -> np.ndarray:
    """Normalize int16 PCM samples into float32 in [-1.0, 1.0].

    Args:
        samples: 1-D int16 numpy array.

    Returns:
        1-D float32 numpy array suitable for sherpa-onnx.
    """
    if samples.size == 0:
        return np.empty(0, dtype=np.float32)
    return samples.astype(np.float32) / 32768.0


class OpenXiaoAiServer:
    """WebSocket server managing a connected Open-XiaoAI speaker client."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 4399,
        *,
        on_audio: AudioCallback | None = None,
        on_event: EventCallback | None = None,
        on_connected: LifecycleCallback | None = None,
        on_disconnected: LifecycleCallback | None = None,
    ) -> None:
        """Initialize the Open-XiaoAI WebSocket server.

        Args:
            host: Bind host address.
            port: Bind TCP port.
            on_audio: Async callback receiving normalized float32 16kHz audio frames.
            on_event: Async callback receiving `(event_name, event_data)` from the speaker.
            on_connected: Async callback invoked after a speaker connects and audio streams start.
            on_disconnected: Async callback invoked when the active speaker disconnects.
        """
        self.host = host
        self.port = port
        self.on_audio = on_audio
        self.on_event = on_event
        self.on_connected = on_connected
        self.on_disconnected = on_disconnected
        self._ws: Any | None = None
        self._send_lock = asyncio.Lock()
        self._server: Any | None = None
        self._play_active = False

    @property
    def connected(self) -> bool:
        """Return True if a speaker client is currently connected."""
        return self._ws is not None

    async def start(self) -> None:
        """Start listening for incoming Open-XiaoAI WebSocket connections until cancelled."""
        import websockets

        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            max_size=16 * 1024 * 1024,
        )
        logger.info(f"Open-XiaoAI WebSocket server listening on ws://{self.host}:{self.port}")
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self._server.close()
            await self._server.wait_closed()
            raise

    async def send_rpc(self, command: str, payload: Any = None) -> None:
        """Send an Open-XiaoAI JSON RPC Request frame to the connected speaker.

        Args:
            command: RPC command name (e.g. 'start_recording', 'start_play', 'stop_play', 'run_shell').
            payload: Optional JSON-serializable payload.
        """
        ws = self._ws
        if ws is None:
            return
        req = {
            "Request": {
                "id": str(uuid.uuid4()),
                "command": command,
                "payload": payload,
            }
        }
        async with self._send_lock:
            if self._ws is ws:
                await ws.send(json.dumps(req))

    async def start_recording(self) -> None:
        """Instruct the speaker to start 16kHz S32_LE microphone capture."""
        await self.send_rpc(
            "start_recording",
            {
                "pcm": "noop",
                "channels": 1,
                "bits_per_sample": 32,
                "sample_rate": RECORD_SAMPLE_RATE,
                "period_size": 360,
                "buffer_size": 1440,
            },
        )

    async def start_play(self, sample_rate: int = PLAY_SAMPLE_RATE) -> None:
        """Instruct the speaker to open its `aplay` raw S16_LE playback stream.

        Args:
            sample_rate: Playback sample rate in Hz (default 24000).
        """
        await self.send_rpc(
            "start_play",
            {
                "pcm": "noop",
                "channels": 1,
                "bits_per_sample": 16,
                "sample_rate": sample_rate,
                "period_size": 360,
                "buffer_size": 1440,
            },
        )
        self._play_active = True

    async def stop_play(self) -> None:
        """Immediately halt speaker playback and flush the `aplay` audio buffer."""
        self._play_active = False
        await self.send_rpc("stop_play")

    async def play_pcm(self, pcm_bytes: bytes, *, sample_rate: int = PLAY_SAMPLE_RATE) -> None:
        """Stream raw S16_LE mono PCM bytes to the speaker in smooth chunks.

        Args:
            pcm_bytes: Raw 16-bit signed little-endian mono PCM bytes.
            sample_rate: Sample rate in Hz (ensures `start_play` is active).
        """
        ws = self._ws
        if ws is None or not pcm_bytes:
            return
        if not self._play_active:
            await self.start_play(sample_rate=sample_rate)
        for idx in range(0, len(pcm_bytes), PLAY_CHUNK_BYTES):
            if self._ws is not ws or not self._play_active:
                return
            chunk = pcm_bytes[idx : idx + PLAY_CHUNK_BYTES]
            frame = json.dumps(
                {
                    "id": str(uuid.uuid4()),
                    "tag": "play",
                    "bytes": list(chunk),
                },
                separators=(",", ":"),
            ).encode("utf-8")
            async with self._send_lock:
                if self._ws is ws and self._play_active:
                    await ws.send(frame)
            await asyncio.sleep(0)

    async def run_shell(self, script: str) -> None:
        """Execute a shell command on the speaker via Open-XiaoAI `run_shell` RPC.

        Args:
            script: POSIX shell command string to run on the LX06 speaker.
        """
        await self.send_rpc("run_shell", script)

    async def abort_native_xiaoai(self) -> None:
        """Interrupt native XiaoAi speech/dialog service so it does not talk over Kesoku."""
        await self.run_shell("/etc/init.d/mico_aivs_lab restart >/dev/null 2>&1")

    async def _handle_connection(self, ws: Any) -> None:
        """Handle one speaker WebSocket lifecycle."""
        if self._ws is not None:
            logger.warning("Replacing existing Open-XiaoAI speaker connection with new client")
            try:
                await self._ws.close()
            except Exception:
                pass

        self._ws = ws
        self._play_active = False
        logger.info(f"Open-XiaoAI speaker connected from {ws.remote_address}")

        try:
            await asyncio.sleep(0.3)
            await self.start_recording()
            await self.start_play(sample_rate=PLAY_SAMPLE_RATE)
            if self.on_connected is not None:
                await self.on_connected()

            async for raw in ws:
                if isinstance(raw, bytes):
                    await self._handle_binary_frame(raw)
                else:
                    await self._handle_text_frame(ws, raw)
        except Exception as e:
            logger.warning(f"Open-XiaoAI speaker connection closed: {e}")
        finally:
            if self._ws is ws:
                self._ws = None
                self._play_active = False
                if self.on_disconnected is not None:
                    try:
                        await self.on_disconnected()
                    except Exception as cb_err:
                        logger.warning(f"XiaoAi disconnect callback error: {cb_err}")
            logger.info("Open-XiaoAI speaker disconnected")

    async def _handle_binary_frame(self, raw: bytes) -> None:
        """Decode an incoming Open-XiaoAI binary Stream frame and dispatch audio."""
        if self.on_audio is None:
            return
        try:
            stream = json.loads(raw)
        except Exception:
            return
        if stream.get("tag") != "record":
            return
        raw_audio = bytes(stream.get("bytes", ()))
        if not raw_audio:
            return
        pcm16 = decode_a113_s32_to_s16(raw_audio)
        if pcm16.size > 0:
            await self.on_audio(s16_to_float32(pcm16))

    async def _handle_text_frame(self, ws: Any, raw: str) -> None:
        """Handle an incoming JSON Request/Response/Event text frame."""
        try:
            msg = json.loads(raw)
        except Exception:
            return
        if "Request" in msg:
            req = msg["Request"]
            if req.get("command") == "get_version":
                resp = {"Response": {"id": req["id"], "data": "1.0.0"}}
                async with self._send_lock:
                    if self._ws is ws:
                        await ws.send(json.dumps(resp))
        elif "Event" in msg and self.on_event is not None:
            ev = msg["Event"]
            await self.on_event(str(ev.get("event", "")), ev.get("data"))
