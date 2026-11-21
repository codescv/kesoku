"""TTS synthesis, streaming sentence chunking, and chime generation for XiaoAi speaker."""

from __future__ import annotations

import asyncio
import math
import random
import re
import subprocess
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from kesoku.logger import setup_logger

from .ws_server import PLAY_SAMPLE_RATE

logger = setup_logger(__name__)

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_TAG_RE = re.compile(r"\[(file|voice|question):[^\]]+\]")
_HTML_TAG_RE = re.compile(r"</?[^>]+>")
_MD_SYMBOLS_RE = re.compile(r"[*_~#>]+")
_MULTI_SPACE_RE = re.compile(r"\s+")

_HARD_BREAK_CHARS = frozenset("。！？!?；;\n")
_SOFT_BREAK_CHARS = frozenset("，,、：:")


def clean_text_for_speech(text: str) -> str:
    """Strip Markdown, code blocks, and Kesoku media tags for natural speech output.

    Args:
        text: Raw assistant response text.

    Returns:
        Plain spoken-friendly text.
    """
    cleaned = _CODE_BLOCK_RE.sub("，代码内容请在屏幕查看。", text)
    cleaned = _TAG_RE.sub("", cleaned)
    cleaned = _MD_LINK_RE.sub(r"\1", cleaned)
    cleaned = _INLINE_CODE_RE.sub(r"\1", cleaned)
    cleaned = _HTML_TAG_RE.sub("", cleaned)
    cleaned = _MD_SYMBOLS_RE.sub("", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()
    return cleaned


class SentenceChunker:
    """Incrementally extracts speakable sentences from streamed accumulated LLM text."""

    def __init__(self, soft_min_chars: int = 18) -> None:
        """Initialize the sentence chunker.

        Args:
            soft_min_chars: Minimum buffer length before splitting on soft punctuation (commas).
        """
        self.soft_min_chars = soft_min_chars
        self._consumed_len = 0
        self._buffer = ""

    def reset(self) -> None:
        """Reset internal state for a new turn."""
        self._consumed_len = 0
        self._buffer = ""

    def feed_accumulated(self, full_text: str) -> list[str]:
        """Feed the latest accumulated response text and emit any completed sentences.

        Args:
            full_text: Full accumulated text emitted so far for the current turn.

        Returns:
            List of newly completed speakable sentences.
        """
        if len(full_text) <= self._consumed_len:
            return []
        delta = full_text[self._consumed_len :]
        self._consumed_len = len(full_text)
        self._buffer += delta
        return self._drain(flush=False)

    def flush(self, final_text: str | None = None) -> list[str]:
        """Flush any remaining buffered text at turn completion.

        Args:
            final_text: Optional final full response text to catch any tail characters.

        Returns:
            List of remaining speakable sentences.
        """
        if final_text and len(final_text) > self._consumed_len:
            self._buffer += final_text[self._consumed_len :]
            self._consumed_len = len(final_text)
        return self._drain(flush=True)

    def _drain(self, *, flush: bool) -> list[str]:
        sentences: list[str] = []
        while self._buffer:
            # Don't split while inside an unclosed fenced code block
            if not flush and self._buffer.count("```") % 2 == 1:
                break

            split_idx = -1
            for i, ch in enumerate(self._buffer):
                if ch in _HARD_BREAK_CHARS:
                    split_idx = i + 1
                    break
                if ch in _SOFT_BREAK_CHARS and i >= self.soft_min_chars:
                    split_idx = i + 1
                    break

            if split_idx == -1:
                if flush:
                    raw = self._buffer
                    self._buffer = ""
                    cleaned = clean_text_for_speech(raw)
                    if cleaned:
                        sentences.append(cleaned)
                break

            raw = self._buffer[:split_idx]
            self._buffer = self._buffer[split_idx:]
            cleaned = clean_text_for_speech(raw)
            if cleaned:
                sentences.append(cleaned)

        return sentences


def make_chime_pcm(
    tones: list[tuple[float, float]],
    *,
    sample_rate: int = PLAY_SAMPLE_RATE,
    volume: float = 0.25,
) -> bytes:
    """Synthesize a crisp multi-tone sine chime as raw S16_LE mono PCM bytes.

    Args:
        tones: List of `(frequency_hz, duration_sec)` pairs.
        sample_rate: Target sample rate in Hz.
        volume: Peak amplitude in (0.0, 1.0].

    Returns:
        Raw S16_LE PCM bytes.
    """
    segments: list[np.ndarray] = []
    for freq, dur in tones:
        n = int(sample_rate * dur)
        t = np.arange(n, dtype=np.float32) / sample_rate
        wave = np.sin(2.0 * math.pi * freq * t)
        # Smooth 10ms fade-in/fade-out envelope to avoid clicks
        fade = min(int(sample_rate * 0.01), n // 2)
        if fade > 0:
            env = np.ones(n, dtype=np.float32)
            env[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            env[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
            wave *= env
        segments.append((wave * volume * 32767.0).astype(np.int16))
    return np.concatenate(segments).tobytes() if segments else b""


def make_wakeup_chime() -> bytes:
    """Return a pleasant two-note ascending wake-up chime (24kHz S16_LE PCM)."""
    return make_chime_pcm([(659.25, 0.08), (880.0, 0.12)], volume=0.28)


def make_sleep_chime() -> bytes:
    """Return a soft descending sleep/end-of-listen chime (24kHz S16_LE PCM)."""
    return make_chime_pcm([(587.33, 0.08), (440.0, 0.12)], volume=0.20)


async def decode_audio_bytes_to_pcm24k(audio_bytes: bytes, input_format: str = "mp3") -> bytes:
    """Transcode compressed audio bytes to 24kHz S16_LE mono PCM via ffmpeg.

    Args:
        audio_bytes: Encoded audio bytes (e.g. MP3 from Edge-TTS).
        input_format: FFmpeg input container hint (e.g. 'mp3', 'wav').

    Returns:
        Raw 24kHz S16_LE mono PCM bytes.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        input_format,
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(PLAY_SAMPLE_RATE),
        "-ac",
        "1",
        "pipe:1",
    ]
    proc = await asyncio.to_thread(
        subprocess.run,
        cmd,
        input=audio_bytes,
        capture_output=True,
        check=True,
    )
    return proc.stdout


async def decode_audio_file_to_pcm24k(file_path: str) -> bytes:
    """Transcode an audio file on disk to 24kHz S16_LE mono PCM via ffmpeg.

    Args:
        file_path: Path to audio file (wav, mp3, ogg, etc.).

    Returns:
        Raw 24kHz S16_LE mono PCM bytes.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        file_path,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(PLAY_SAMPLE_RATE),
        "-ac",
        "1",
        "pipe:1",
    ]
    proc = await asyncio.to_thread(
        subprocess.run,
        cmd,
        capture_output=True,
        check=True,
    )
    return proc.stdout


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Clip float32 samples in [-1.0, 1.0] and convert to little-endian int16 PCM bytes.

    Args:
        audio: 1-D numpy float waveform array.

    Returns:
        Raw S16_LE PCM bytes.
    """
    if audio.size == 0:
        return b""
    flat = np.asarray(audio, dtype=np.float32).reshape(-1)
    clipped = np.clip(flat, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def trim_pcm16_silence(pcm: bytes, *, threshold: int = 300, pad_samples: int = 720) -> bytes:
    """Trim leading and trailing silence from S16_LE PCM bytes.

    Args:
        pcm: Raw little-endian int16 mono PCM bytes.
        threshold: Minimum absolute sample amplitude treated as speech.
        pad_samples: Padding samples retained on both ends (~30ms at 24kHz).

    Returns:
        Trimmed raw S16_LE PCM bytes, or empty bytes if fully silent.
    """
    if len(pcm) < 2:
        return b""
    samples = np.frombuffer(pcm[: (len(pcm) // 2) * 2], dtype="<i2")
    active = np.flatnonzero(np.abs(samples.astype(np.int32)) >= threshold)
    if active.size == 0:
        return b""
    start = max(0, int(active[0]) - pad_samples)
    end = min(samples.size, int(active[-1]) + 1 + pad_samples)
    return samples[start:end].tobytes()


def _scan_cue_files(root: Path) -> dict[str, list[Path]] | None:
    """Scan `root` for category subdirectories containing `.wav`/`.mp3` cue files."""
    if not root.is_dir():
        return None
    grouped: dict[str, list[Path]] = {}
    for cat_dir in sorted(root.iterdir()):
        if not cat_dir.is_dir():
            continue
        files = [
            f
            for f in sorted(cat_dir.iterdir())
            if f.suffix.lower() in {".wav", ".mp3"} and f.is_file()
        ]
        if files:
            grouped[cat_dir.name] = files
    return grouped


class VoiceCueBank:
    """In-memory bank of pre-rendered 24kHz S16_LE voice cues grouped by category."""

    def __init__(self) -> None:
        """Initialize an empty cue bank."""
        self._clips: dict[str, list[bytes]] = {}
        self._last_idx: dict[str, int] = {}

    async def load_dir(self, cues_dir: str | Path) -> int:
        """Load and silence-trim `.wav`/`.mp3` clips from category subdirectories.

        Args:
            cues_dir: Root cues directory containing category subdirectories (e.g. `wakeup/`).

        Returns:
            Total number of audio clips loaded.
        """
        root = Path(cues_dir)
        self._clips.clear()
        self._last_idx.clear()
        grouped = await asyncio.to_thread(_scan_cue_files, root)
        if grouped is None:
            logger.warning(f"XiaoAi cues directory not found: {root}")
            return 0

        total = 0
        for category, files in grouped.items():
            clips: list[bytes] = []
            for audio_file in files:
                try:
                    raw_pcm = await decode_audio_file_to_pcm24k(str(audio_file))
                    trimmed = trim_pcm16_silence(raw_pcm)
                    if trimmed:
                        clips.append(trimmed)
                except Exception as e:
                    logger.warning(f"Failed to load cue '{audio_file}': {e}")
            if clips:
                self._clips[category] = clips
                total += len(clips)
        return total

    def counts(self) -> dict[str, int]:
        """Return loaded clip counts per category."""
        return {k: len(v) for k, v in self._clips.items()}

    def pick(self, category: str) -> bytes | None:
        """Pick a random cue clip from `category`, avoiding immediate repeats.

        Args:
            category: Cue category directory name (e.g. 'wakeup', 'tool_search').

        Returns:
            24kHz S16_LE PCM bytes, or None if no clips exist for `category`.
        """
        clips = self._clips.get(category)
        if not clips:
            return None
        if len(clips) == 1:
            return clips[0]
        last = self._last_idx.get(category, -1)
        candidates = [i for i in range(len(clips)) if i != last]
        idx = random.choice(candidates)
        self._last_idx[category] = idx
        return clips[idx]


class TTSEngine(Protocol):
    """Protocol for pluggable smart speaker TTS backends."""

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text into 24kHz S16_LE mono PCM bytes."""
        ...


class EdgeTTSEngine:
    """Low-latency Edge-TTS backend producing 24kHz S16_LE mono PCM."""

    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural", rate: str = "+0%") -> None:
        """Initialize the Edge-TTS engine.

        Args:
            voice: Microsoft Edge neural voice name.
            rate: Speech rate modifier (e.g. '+0%', '+10%').
        """
        self.voice = voice
        self.rate = rate
        self._cache: dict[str, bytes] = {}

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to 24kHz S16_LE mono PCM bytes.

        Args:
            text: Plain spoken text.

        Returns:
            Raw 24kHz S16_LE mono PCM bytes.
        """
        import edge_tts

        text = text.strip()
        if not text:
            return b""
        if text in self._cache:
            return self._cache[text]

        communicate = edge_tts.Communicate(text, voice=self.voice, rate=self.rate)
        mp3_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_chunks.append(chunk["data"])
        if not mp3_chunks:
            return b""
        pcm = await decode_audio_bytes_to_pcm24k(b"".join(mp3_chunks), input_format="mp3")
        # Cache very short phrases (tool cues, acknowledgements) for 0ms replay
        if len(text) <= 12 and len(self._cache) < 64:
            self._cache[text] = pcm
        return pcm


def _result_audio_to_pcm24k(audio: Any, sample_rate: int) -> bytes:
    """Convert an MLX audio tensor/array at `sample_rate` into 24kHz S16_LE PCM bytes."""
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return b""
    if sample_rate != PLAY_SAMPLE_RATE and arr.size > 1:
        target_len = max(1, int(round(arr.size * PLAY_SAMPLE_RATE / sample_rate)))
        x_old = np.linspace(0.0, 1.0, arr.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, target_len, endpoint=False)
        arr = np.interp(x_new, x_old, arr).astype(np.float32)
    return float32_to_pcm16(arr)


class MlxQwenTTSEngine:
    """Apple Silicon MLX Qwen3-TTS voice-cloning backend producing 24kHz S16_LE mono PCM."""

    def __init__(
        self,
        model_id: str,
        ref_audio: str | None = None,
        ref_text: str | None = None,
    ) -> None:
        """Initialize the MLX Qwen3-TTS engine.

        Args:
            model_id: HuggingFace MLX model identifier (e.g. 'mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit').
            ref_audio: Optional default filesystem path to reference WAV/audio clip for voice cloning.
            ref_text: Transcript matching `ref_audio`; defaults to filename stem when omitted.
        """
        self.model_id = model_id
        self._voice_specs: dict[str, tuple[str, str]] = {}
        self._voice_arrays: dict[str, tuple[Any, str]] = {}
        self._active_voice = "default"
        self._model: Any | None = None
        self._lock = asyncio.Lock()
        self._cache: dict[tuple[str, str], bytes] = {}
        if ref_audio:
            self.register_voice("default", ref_audio, ref_text)

    @property
    def ref_text(self) -> str:
        """Return the active voice's reference transcript."""
        spec = self._voice_specs.get(self._active_voice)
        return spec[1] if spec is not None else ""

    @property
    def _ref_audio_mx(self) -> Any | None:
        entry = self._voice_arrays.get(self._active_voice)
        return entry[0] if entry is not None else None

    @_ref_audio_mx.setter
    def _ref_audio_mx(self, value: Any) -> None:
        self._voice_arrays[self._active_voice] = (value, self.ref_text)

    def register_voice(self, voice_id: str, ref_audio: str, ref_text: str | None = None) -> None:
        """Register a named voice profile backed by a reference audio clip.

        Args:
            voice_id: Persona/voice identifier (e.g. 'leijun', 'xiaozhang').
            ref_audio: Filesystem path to reference audio file.
            ref_text: Transcript matching `ref_audio`; defaults to filename stem when omitted.
        """
        resolved_text = (ref_text or "").strip() or Path(ref_audio).stem
        self._voice_specs[voice_id] = (ref_audio, resolved_text)
        if not self._voice_specs or self._active_voice not in self._voice_specs:
            self._active_voice = voice_id
        if self._model is not None:
            from mlx_audio.tts.models.qwen3_tts.qwen3_tts import load_audio

            model_sr = int(getattr(self._model, "sample_rate", PLAY_SAMPLE_RATE))
            self._voice_arrays[voice_id] = (load_audio(str(ref_audio), sample_rate=model_sr), resolved_text)

    def set_active_voice(self, voice_id: str) -> None:
        """Select the default voice profile used when `voice_id` is not explicitly passed.

        Args:
            voice_id: Registered voice profile identifier.
        """
        if voice_id not in self._voice_specs:
            raise KeyError(f"Unregistered MLX Qwen3-TTS voice_id: '{voice_id}'")
        self._active_voice = voice_id

    def load(self) -> None:
        """Load the MLX Qwen3-TTS model, resample all registered voices, and run Metal warmup."""
        if self._model is not None:
            return
        if not self._voice_specs:
            raise ValueError("No reference audio voices registered for MlxQwenTTSEngine")
        for vid, (raw_path, _) in self._voice_specs.items():
            if not Path(raw_path).exists():
                raise FileNotFoundError(f"Reference audio file not found for voice '{vid}': {raw_path}")

        from mlx_audio.tts.models.qwen3_tts.qwen3_tts import load_audio
        from mlx_audio.tts.utils import load_model

        logger.info(
            f"Loading MLX Qwen3-TTS model '{self.model_id}' for voices {list(self._voice_specs.keys())}..."
        )
        t0 = time.perf_counter()
        model = load_model(self.model_id)
        model_sr = int(getattr(model, "sample_rate", PLAY_SAMPLE_RATE))
        for vid, (raw_path, rtext) in self._voice_specs.items():
            ref_mx = load_audio(str(raw_path), sample_rate=model_sr)
            self._voice_arrays[vid] = (ref_mx, rtext)
        self._model = model
        logger.info(
            f"Loaded MLX Qwen3-TTS model in {time.perf_counter() - t0:.2f}s "
            f"(sample_rate={model_sr}, voices={list(self._voice_arrays.keys())})"
        )
        # Compile Metal kernels before the first user wake-up
        self._synthesize_sync("你好", voice_id=self._active_voice)

    def _resolve_voice(self, voice_id: str | None) -> tuple[str, Any, str]:
        """Resolve `(voice_id, ref_audio_mx, ref_text)` for synthesis."""
        vid = voice_id or self._active_voice
        if self._model is None:
            self.load()
        entry = self._voice_arrays.get(vid)
        if entry is None:
            if vid not in self._voice_specs:
                raise KeyError(f"Unregistered MLX Qwen3-TTS voice_id: '{vid}'")
            from mlx_audio.tts.models.qwen3_tts.qwen3_tts import load_audio

            raw_path, rtext = self._voice_specs[vid]
            model_sr = int(getattr(self._model, "sample_rate", PLAY_SAMPLE_RATE))
            entry = (load_audio(str(raw_path), sample_rate=model_sr), rtext)
            self._voice_arrays[vid] = entry
        return vid, entry[0], entry[1]

    def _synthesize_sync(self, text: str, voice_id: str | None = None) -> bytes:
        """Synthesize text synchronously on Apple Silicon MLX and return 24kHz S16_LE PCM bytes."""
        vid, ref_mx, ref_text = self._resolve_voice(voice_id)
        assert self._model is not None

        t0 = time.perf_counter()
        results = list(
            self._model.generate(
                text=text,
                ref_audio=ref_mx,
                ref_text=ref_text,
                lang_code="auto",
                split_pattern="",
                verbose=False,
            )
        )
        if not results:
            return b""

        pcm_parts = [
            _result_audio_to_pcm24k(res.audio, int(getattr(res, "sample_rate", PLAY_SAMPLE_RATE)))
            for res in results
        ]
        pcm = b"".join(pcm_parts)
        if not pcm:
            return b""

        synth_sec = time.perf_counter() - t0
        audio_sec = len(pcm) / (PLAY_SAMPLE_RATE * 2)
        rtf = (synth_sec / audio_sec) if audio_sec > 0 else 0.0
        logger.info(
            f"MLX Qwen3-TTS synthesis complete: voice={vid} chars={len(text)} "
            f"audio_sec={audio_sec:.2f}s synth_sec={synth_sec:.2f}s rtf={rtf:.2f}"
        )
        return pcm

    async def synthesize(self, text: str, *, voice_id: str | None = None) -> bytes:
        """Synthesize text to 24kHz S16_LE mono PCM bytes without blocking the event loop.

        Args:
            text: Plain spoken text.
            voice_id: Optional registered voice profile identifier.

        Returns:
            Raw 24kHz S16_LE mono PCM bytes.
        """
        text = text.strip()
        if not text:
            return b""
        vid = voice_id or self._active_voice
        cache_key = (vid, text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        async with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
            if voice_id is None:
                pcm = await asyncio.to_thread(self._synthesize_sync, text)
            else:
                pcm = await asyncio.to_thread(self._synthesize_sync, text, voice_id)

        if pcm and len(text) <= 12 and len(self._cache) < 64:
            self._cache[cache_key] = pcm
        return pcm

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice_id: str | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream 24kHz S16_LE PCM audio chunks progressively as MLX generates them.

        Args:
            text: Plain spoken text.
            voice_id: Optional registered voice profile identifier.
            is_cancelled: Optional zero-arg predicate; stops MLX generation early when True.

        Yields:
            24kHz S16_LE mono PCM audio chunks.
        """
        text = text.strip()
        if not text:
            return
        vid = voice_id or self._active_voice
        cache_key = (vid, text)
        if cache_key in self._cache:
            yield self._cache[cache_key]
            return

        loop = asyncio.get_running_loop()
        chunk_queue: asyncio.Queue[bytes | None | Exception] = asyncio.Queue()
        collected: list[bytes] = []
        cancelled_early = False

        def _worker() -> None:
            nonlocal cancelled_early
            try:
                resolved_vid, ref_mx, ref_text = self._resolve_voice(vid)
                assert self._model is not None

                t0 = time.perf_counter()
                ttfa_sec: float | None = None
                total_bytes = 0
                for res in self._model.generate(
                    text=text,
                    ref_audio=ref_mx,
                    ref_text=ref_text,
                    lang_code="auto",
                    split_pattern="",
                    stream=True,
                    streaming_interval=1.0,
                    verbose=False,
                ):
                    if is_cancelled is not None and is_cancelled():
                        cancelled_early = True
                        break
                    sr = int(getattr(res, "sample_rate", PLAY_SAMPLE_RATE))
                    pcm = _result_audio_to_pcm24k(res.audio, sr)
                    if not pcm:
                        continue
                    if ttfa_sec is None:
                        ttfa_sec = time.perf_counter() - t0
                    total_bytes += len(pcm)
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, pcm)

                synth_sec = time.perf_counter() - t0
                audio_sec = total_bytes / (PLAY_SAMPLE_RATE * 2)
                rtf = (synth_sec / audio_sec) if audio_sec > 0 else 0.0
                logger.info(
                    f"MLX Qwen3-TTS stream complete: voice={resolved_vid} chars={len(text)} "
                    f"ttfa={ttfa_sec or 0.0:.2f}s audio_sec={audio_sec:.2f}s "
                    f"synth_sec={synth_sec:.2f}s rtf={rtf:.2f} cancelled={cancelled_early}"
                )
            except Exception as exc:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, None)

        async with self._lock:
            if cache_key in self._cache:
                yield self._cache[cache_key]
                return
            worker_task = asyncio.create_task(asyncio.to_thread(_worker))
            try:
                while True:
                    item = await chunk_queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    if is_cancelled is not None and is_cancelled():
                        cancelled_early = True
                        break
                    collected.append(item)
                    yield item
            finally:
                await worker_task

        if collected and not cancelled_early and (is_cancelled is None or not is_cancelled()):
            full_pcm = b"".join(collected)
            if len(text) <= 12 and len(self._cache) < 64:
                self._cache[cache_key] = full_pcm
