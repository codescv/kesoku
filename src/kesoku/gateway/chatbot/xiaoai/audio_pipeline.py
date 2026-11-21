"""Local sherpa-onnx KWS (wake word), Silero VAD, and SenseVoiceSmall ASR pipeline."""

from __future__ import annotations

import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from kesoku.logger import setup_logger

logger = setup_logger(__name__)

SAMPLE_RATE = 16000
VAD_WINDOW_SIZE = 512

KWS_MODEL_NAME = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
KWS_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
    f"{KWS_MODEL_NAME}.tar.bz2"
)

VAD_MODEL_NAME = "silero_vad.onnx"
VAD_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{VAD_MODEL_NAME}"
)

ASR_MODEL_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
ASR_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{ASR_MODEL_NAME}.tar.bz2"
)


def _download_file(url: str, dest: Path) -> None:
    """Download a file to `dest` atomically via a temporary `.part` file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, tmp)  # noqa: S310
    tmp.replace(dest)


def _ensure_tar_model(models_dir: Path, name: str, url: str) -> Path:
    """Ensure a tar.bz2 sherpa-onnx model directory exists under `models_dir`."""
    target_dir = models_dir / name
    if target_dir.is_dir():
        return target_dir
    archive = models_dir / f"{name}.tar.bz2"
    if not archive.exists():
        _download_file(url, archive)
    logger.info(f"Extracting {archive} -> {models_dir}")
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(path=models_dir, filter="data")
    archive.unlink(missing_ok=True)
    return target_dir


def _ensure_file_model(models_dir: Path, name: str, url: str) -> Path:
    """Ensure a standalone ONNX model file exists under `models_dir`."""
    target = models_dir / name
    if not target.exists():
        _download_file(url, target)
    return target


def chinese_phrase_to_kws_tokens(phrase: str, valid_tokens: set[str] | None = None) -> str:
    """Convert a Chinese wake word phrase into sherpa-onnx WenetSpeech KWS token format.

    Example:
        `"小张小张"` -> `"x iǎo zh āng x iǎo zh āng @小张小张"`

    Args:
        phrase: Chinese wake word phrase (or raw token string if it already contains `@`).
        valid_tokens: Optional set of tokens loaded from `tokens.txt` for validation.

    Returns:
        Formatted `keywords.txt` line.
    """
    stripped = phrase.strip()
    if "@" in stripped:
        return stripped

    from pypinyin import Style, pinyin

    initials = pinyin(stripped, style=Style.INITIALS, strict=False)
    finals = pinyin(stripped, style=Style.FINALS_TONE, strict=False)

    tokens: list[str] = []
    for init_list, fin_list in zip(initials, finals, strict=False):
        init = init_list[0].strip() if init_list else ""
        fin = fin_list[0].strip() if fin_list else ""
        if init:
            tokens.append(init)
        if fin:
            tokens.append(fin)

    if valid_tokens:
        missing = [t for t in tokens if t not in valid_tokens]
        if missing:
            logger.warning(f"KWS wake word '{stripped}' has unknown tokens {missing}; detection may be degraded")

    return f"{' '.join(tokens)} @{stripped}"


class AudioPipeline:
    """Synchronous CPU-bound sherpa-onnx KWS + Silero VAD + SenseVoiceSmall ASR wrapper."""

    def __init__(
        self,
        models_dir: str | Path,
        wake_words: list[str],
        *,
        kws_score: float = 1.5,
        kws_threshold: float = 0.25,
        vad_silence_sec: float = 0.7,
        num_threads: int = 2,
    ) -> None:
        """Initialize pipeline parameters (call `load()` before feeding audio).

        Args:
            models_dir: Directory where sherpa-onnx models are stored/downloaded.
            wake_words: List of custom wake word phrases (e.g. `["小张小张"]`).
            kws_score: Keyword spotting boosting score.
            kws_threshold: Keyword spotting trigger threshold.
            vad_silence_sec: Minimum trailing silence duration to finalize an utterance.
            num_threads: ONNX Runtime CPU thread count.
        """
        self.models_dir = Path(models_dir)
        self.wake_words = wake_words or ["小张小张"]
        self.kws_score = kws_score
        self.kws_threshold = kws_threshold
        self.vad_silence_sec = vad_silence_sec
        self.num_threads = num_threads

        self._kws: Any | None = None
        self._kws_stream: Any | None = None
        self._vad: Any | None = None
        self._vad_buf = np.empty(0, dtype=np.float32)
        self._asr: Any | None = None

    def load(self) -> None:
        """Download (if missing) and initialize KWS, VAD, and ASR models."""
        import sherpa_onnx

        self.models_dir.mkdir(parents=True, exist_ok=True)
        kws_dir = _ensure_tar_model(self.models_dir, KWS_MODEL_NAME, KWS_MODEL_URL)
        vad_path = _ensure_file_model(self.models_dir, VAD_MODEL_NAME, VAD_MODEL_URL)
        asr_dir = _ensure_tar_model(self.models_dir, ASR_MODEL_NAME, ASR_MODEL_URL)

        # Build keywords.txt from configured wake_words
        tokens_path = kws_dir / "tokens.txt"
        valid_tokens = {
            line.split()[0]
            for line in tokens_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        kw_lines = [chinese_phrase_to_kws_tokens(w, valid_tokens) for w in self.wake_words]
        kw_file = self.models_dir / "keywords.txt"
        kw_file.write_text("\n".join(kw_lines) + "\n", encoding="utf-8")
        logger.info(f"Prepared KWS keywords ({kw_file}): {kw_lines}")

        encoder = next(kws_dir.glob("encoder-*.int8.onnx"), next(kws_dir.glob("encoder-*.onnx")))
        decoder = next(kws_dir.glob("decoder-*.int8.onnx"), next(kws_dir.glob("decoder-*.onnx")))
        joiner = next(kws_dir.glob("joiner-*.int8.onnx"), next(kws_dir.glob("joiner-*.onnx")))

        self._kws = sherpa_onnx.KeywordSpotter(
            tokens=str(tokens_path),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            num_threads=self.num_threads,
            keywords_file=str(kw_file),
            keywords_score=self.kws_score,
            keywords_threshold=self.kws_threshold,
            provider="cpu",
        )
        self._kws_stream = self._kws.create_stream()

        vad_cfg = sherpa_onnx.VadModelConfig()
        vad_cfg.silero_vad.model = str(vad_path)
        vad_cfg.silero_vad.threshold = 0.5
        vad_cfg.silero_vad.min_silence_duration = self.vad_silence_sec
        vad_cfg.silero_vad.min_speech_duration = 0.25
        vad_cfg.silero_vad.window_size = VAD_WINDOW_SIZE
        vad_cfg.sample_rate = SAMPLE_RATE
        self._vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=30)

        asr_model = asr_dir / "model.int8.onnx"
        if not asr_model.exists():
            asr_model = asr_dir / "model.onnx"
        self._asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(asr_model),
            tokens=str(asr_dir / "tokens.txt"),
            num_threads=max(2, self.num_threads),
            use_itn=True,
            language="auto",
            provider="cpu",
        )
        logger.info("Loaded sherpa-onnx KWS + Silero VAD + SenseVoiceSmall ASR models successfully")

    def reset_vad(self) -> None:
        """Clear buffered VAD samples and pending segments."""
        self._vad_buf = np.empty(0, dtype=np.float32)
        if self._vad is not None:
            self._vad.reset()

    def reset_kws(self) -> None:
        """Reset the KWS stream state."""
        if self._kws is not None and self._kws_stream is not None:
            self._kws.reset_stream(self._kws_stream)

    def feed_kws(self, samples: np.ndarray) -> str | None:
        """Feed 16kHz float32 audio into the keyword spotter.

        Args:
            samples: 1-D float32 numpy array at 16kHz.

        Returns:
            Detected wake word string if triggered, otherwise None.
        """
        if self._kws is None or self._kws_stream is None or samples.size == 0:
            return None
        self._kws_stream.accept_waveform(SAMPLE_RATE, samples)
        while self._kws.is_ready(self._kws_stream):
            self._kws.decode_stream(self._kws_stream)
        keyword = self._kws.get_result(self._kws_stream).strip()
        if keyword:
            self._kws.reset_stream(self._kws_stream)
            return keyword
        return None

    def feed_vad(self, samples: np.ndarray) -> tuple[bool, list[np.ndarray]]:
        """Feed 16kHz float32 audio into Silero VAD and collect completed speech segments.

        Args:
            samples: 1-D float32 numpy array at 16kHz.

        Returns:
            Tuple of `(is_speech_detected, completed_segments)`.
        """
        if self._vad is None or samples.size == 0:
            return False, []
        self._vad_buf = np.concatenate([self._vad_buf, samples])
        while self._vad_buf.size >= VAD_WINDOW_SIZE:
            window = self._vad_buf[:VAD_WINDOW_SIZE]
            self._vad_buf = self._vad_buf[VAD_WINDOW_SIZE:]
            self._vad.accept_waveform(window)

        is_speech = bool(self._vad.is_speech_detected())
        segments: list[np.ndarray] = []
        while not self._vad.empty():
            seg = np.asarray(self._vad.front.samples, dtype=np.float32)
            self._vad.pop()
            if seg.size >= int(SAMPLE_RATE * 0.25):
                segments.append(seg)
        return is_speech, segments

    def transcribe(self, samples: np.ndarray) -> str:
        """Transcribe a completed 16kHz float32 speech utterance with SenseVoiceSmall.

        Args:
            samples: 1-D float32 numpy array at 16kHz.

        Returns:
            Recognized text string (stripped).
        """
        if self._asr is None or samples.size == 0:
            return ""
        stream = self._asr.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._asr.decode_stream(stream)
        return str(stream.result.text).strip()

    @staticmethod
    def resolve_models_dir(raw_path: str, base_dir: str | None) -> Path:
        """Resolve `models_dir` relative to the agent working directory if not absolute.

        Args:
            raw_path: Configured `models_dir` string.
            base_dir: Agent working directory path, or None.

        Returns:
            Resolved `Path`.
        """
        p = Path(os.path.expanduser(raw_path))
        if not p.is_absolute() and base_dir:
            p = Path(base_dir) / p
        return p
