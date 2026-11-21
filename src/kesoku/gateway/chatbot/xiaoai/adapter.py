"""XiaoAi smart speaker chatbot adapter for Kesoku AI Agent framework."""

from __future__ import annotations

import asyncio
import enum
import os
import time
import uuid
from typing import Any

import numpy as np

from kesoku.config import XiaoAiConfig, XiaoAiPersonaConfig, get_config
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.chatbot.base import Chatbot, InboundMessageDTO
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger
from kesoku.utils.async_fs import async_exists, async_read_text_file

from .audio_pipeline import AudioPipeline
from .tts import (
    EdgeTTSEngine,
    MlxQwenTTSEngine,
    SentenceChunker,
    TTSEngine,
    VoiceCueBank,
    decode_audio_file_to_pcm24k,
    make_sleep_chime,
    make_wakeup_chime,
)
from .ws_server import OpenXiaoAiServer

logger = setup_logger(__name__)

SPEAKER_CHANNEL_ID = "xiaoai:default"
SPEAKER_SENDER_ID = "xiaoai_user"

_TOOL_VOICE_HINTS: dict[str, str] = {
    "web_search": "好的，我帮你查一下。",
    "run_shell_command": "稍等，我正在处理。",
    "read_file": "我先看一下相关内容。",
    "memory_search": "我想一下。",
    "use_skill": "好的，正在为你调用工具。",
    "delegate_subagent": "稍等，我正在分析。",
}
_DEFAULT_TOOL_VOICE_HINT = "稍等一下，我正在处理。"
_SEARCH_TOOLS = frozenset(
    {
        "web_search",
        "web_fetch",
        "read_file",
        "memory_search",
        "session_search",
        "search_memory",
        "list_files",
        "grep_search",
    }
)

_KWS_STOP_WORDS = ("闭嘴", "停下", "别说了", "退下", "停止播放")
_CORE_STOP_SUBSTRINGS = ("闭嘴", "停下", "别说了", "不要说了", "别讲了", "退下", "停止播放")
_STOP_COMMANDS = frozenset(
    {
        "停下",
        "停止",
        "闭嘴",
        "闭嘴吧",
        "停一下",
        "别说了",
        "不要说了",
        "别讲了",
        "退下",
        "好了",
        "行了",
        "停",
        "静音",
        "停止播放",
    }
)
_CLEAR_COMMANDS = frozenset(
    {
        "新对话",
        "开新对话",
        "开启新对话",
        "清空对话",
        "重新开始",
        "重新对话",
        "换个话题",
        "忘记刚才的",
    }
)
_CORE_CLEAR_SUBSTRINGS = ("新对话", "清空对话", "重新开始", "重新对话", "换个话题")
_VOL_UP_COMMANDS = frozenset({"大声点", "声音大一点", "音量大一点", "大点声"})
_VOL_DOWN_COMMANDS = frozenset({"小声点", "声音小一点", "音量小一点", "小点声"})


class SpeakerState(enum.StrEnum):
    """Voice interaction state machine states for the smart speaker."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


def _resolve_awd_path(raw_path: str, awd: str | None) -> str:
    """Resolve `raw_path` relative to `awd` when not already absolute."""
    expanded = os.path.expanduser(raw_path)
    if not os.path.isabs(expanded) and awd:
        return os.path.join(awd, expanded)
    return expanded


class XiaoAiChatbot(Chatbot):
    """Smart speaker adapter connecting an Open-XiaoAI LX06 speaker to Kesoku."""

    def __init__(
        self,
        chatbot_id: str,
        gateway: Gateway,
        xiaoai_config: XiaoAiConfig,
    ) -> None:
        """Initialize the XiaoAi smart speaker adapter.

        Args:
            chatbot_id: Unique chatbot identifier (default 'xiaoai').
            gateway: Kesoku Gateway broker.
            xiaoai_config: `[[chatbots.xiaoai]]` configuration.
        """
        self.stream_min_interval_sec = float(xiaoai_config.stream_min_interval_sec)
        super().__init__(chatbot_id, gateway)
        self.config = xiaoai_config
        self.state = SpeakerState.IDLE
        self._listen_deadline = 0.0
        self._wakeup_chime = make_wakeup_chime()
        self._sleep_chime = make_sleep_chime()

        cfg = get_config()
        awd = cfg.agent_working_dir

        persona_list = xiaoai_config.personas or [
            XiaoAiPersonaConfig(
                role=xiaoai_config.default_role,
                wake_words=list(xiaoai_config.wake_words),
                qwen_tts_ref_audio=xiaoai_config.qwen_tts_ref_audio,
                qwen_tts_ref_text=xiaoai_config.qwen_tts_ref_text,
                tts_voice=xiaoai_config.tts_voice,
                cues_dir=xiaoai_config.cues_dir,
            )
        ]
        self._personas: dict[str, XiaoAiPersonaConfig] = {}
        self._wake_word_to_role: dict[str, str] = {}
        self._all_wake_words: list[str] = []
        self._persona_cues: dict[str, VoiceCueBank] = {}
        self._persona_cues_dirs: dict[str, str] = {}
        for persona in persona_list:
            self._personas[persona.role] = persona
            self._persona_cues[persona.role] = VoiceCueBank()
            if persona.cues_dir:
                self._persona_cues_dirs[persona.role] = _resolve_awd_path(persona.cues_dir, awd)
            for ww in persona.wake_words:
                self._wake_word_to_role[ww] = persona.role
                if ww not in self._all_wake_words:
                    self._all_wake_words.append(ww)

        self._active_role = (
            xiaoai_config.default_role
            if xiaoai_config.default_role in self._personas
            else next(iter(self._personas))
        )

        models_dir = AudioPipeline.resolve_models_dir(xiaoai_config.models_dir, awd)
        self.pipeline = AudioPipeline(
            models_dir=models_dir,
            wake_words=self._all_wake_words,
            stop_words=list(_KWS_STOP_WORDS),
            kws_score=xiaoai_config.kws_score,
            kws_threshold=xiaoai_config.kws_threshold,
            vad_silence_sec=xiaoai_config.vad_silence_sec,
        )
        self.tts: TTSEngine
        if xiaoai_config.tts_provider == "qwen_tts":
            qwen_engine = MlxQwenTTSEngine(model_id=xiaoai_config.qwen_tts_model)
            for role, persona in self._personas.items():
                raw_ref = persona.qwen_tts_ref_audio or xiaoai_config.qwen_tts_ref_audio
                if not raw_ref:
                    raise ValueError(
                        f"qwen_tts_ref_audio must be configured for persona '{role}' when tts_provider='qwen_tts'"
                    )
                ref_path = _resolve_awd_path(raw_ref, awd)
                ref_text = persona.qwen_tts_ref_text or xiaoai_config.qwen_tts_ref_text
                qwen_engine.register_voice(role, ref_path, ref_text)
            qwen_engine.set_active_voice(self._active_role)
            self.tts = qwen_engine
        else:
            self.tts = EdgeTTSEngine(voice=xiaoai_config.tts_voice, rate=xiaoai_config.tts_rate)
        self.ws_server = OpenXiaoAiServer(
            host=xiaoai_config.host,
            port=xiaoai_config.port,
            on_audio=self._on_audio_frame,
            on_event=self._on_speaker_event,
            on_connected=self._on_speaker_connected,
            on_disconnected=self._on_speaker_disconnected,
            auto_start_recording=not xiaoai_config.local_kws,
            on_local_wakeup=self._on_local_wakeup,
            on_mic_mute=self._on_mic_mute,
        )

        self._chunker = SentenceChunker()
        self._active_stream_turn_id: str | None = None
        self._streamed_any_sentence = False
        self._tts_queue: asyncio.Queue[tuple[int, str, str, bytes | None]] = asyncio.Queue()
        self._play_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(maxsize=32)
        self._tts_generation = 0
        self._synth_in_flight = False
        self._play_end_ts = 0.0
        self._synth_worker_task: asyncio.Task[None] | None = None
        self._play_worker_task: asyncio.Task[None] | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._local_wakeup_task: asyncio.Task[None] | None = None
        self._mic_mute_task: asyncio.Task[None] | None = None
        self._mic_muted = False
        self._last_tool_hint_ts = 0.0
        self._turn_start_ts = 0.0
        self._vad_unmute_ts = 0.0
        self._turn_in_progress = False
        self._suppress_until_next_user_turn = False
        self._active_turn_id: str | None = None
        self._active_turn_channel_id: str | None = None
        self._aborted_turn_ids: set[str] = set()
        self._audio_lock = asyncio.Lock()

        self.gateway.register_stream_handler(self.chatbot_id, self._handle_stream_update)

    def _active_channel_id(self) -> str:
        """Return the isolated channel ID for the active persona (or default for single persona)."""
        personas = getattr(self, "_personas", None)
        if personas and len(personas) > 1:
            return f"xiaoai:{self._active_role}"
        return SPEAKER_CHANNEL_ID

    def _all_channel_ids(self) -> list[str]:
        """Return all distinct speaker channel IDs across configured personas plus fallback."""
        channels: list[str] = []
        personas = getattr(self, "_personas", None)
        if personas and len(personas) > 1:
            channels.extend(f"xiaoai:{role}" for role in personas)
        if SPEAKER_CHANNEL_ID not in channels:
            channels.append(SPEAKER_CHANNEL_ID)
        return channels

    def _record_aborted_turn(self, turn_id: str | None) -> None:
        """Remember an aborted turn ID so late/stale stream chunks and replies are dropped."""
        if not turn_id:
            return
        aborted = getattr(self, "_aborted_turn_ids", None)
        if aborted is None:
            aborted = set()
            self._aborted_turn_ids = aborted
        if len(aborted) >= 128:
            aborted.pop()
        aborted.add(turn_id)

    def _is_stale_turn(self, *, channel_id: str | None, turn_id: str | None) -> bool:
        """Return True if an outbound update/reply belongs to an aborted or superseded turn."""
        aborted = getattr(self, "_aborted_turn_ids", None)
        if turn_id and aborted and turn_id in aborted:
            return True
        active_ch = getattr(self, "_active_turn_channel_id", None)
        if active_ch and channel_id and channel_id != active_ch:
            return True
        active_turn = getattr(self, "_active_turn_id", None)
        if active_turn and turn_id and turn_id != active_turn:
            return True
        return False

    async def start(self) -> None:
        """Load local speech models and start the Open-XiaoAI WS server + Gateway loop."""
        tts_target = (
            self.config.qwen_tts_model
            if self.config.tts_provider == "qwen_tts"
            else self.config.tts_voice
        )
        logger.info(
            f"Starting XiaoAiChatbot '{self.chatbot_id}' on ws://{self.config.host}:{self.config.port} "
            f"(personas={list(self._personas.keys())}, wake_words={self._all_wake_words}, "
            f"local_kws={self.config.local_kws}, tts={self.config.tts_provider}:{tts_target})"
        )
        await asyncio.to_thread(self.pipeline.load)
        if isinstance(self.tts, MlxQwenTTSEngine):
            await asyncio.to_thread(self.tts.load)
        for role, cues_dir in self._persona_cues_dirs.items():
            bank = self._persona_cues[role]
            loaded_cues = await bank.load_dir(cues_dir)
            logger.info(
                f"Loaded {loaded_cues} pre-generated voice cues for persona '{role}' "
                f"from '{cues_dir}': {bank.counts()}"
            )

        self._synth_worker_task = asyncio.create_task(self._synth_worker())
        self._play_worker_task = asyncio.create_task(self._play_worker())
        self._watchdog_task = asyncio.create_task(self._listen_watchdog())

        ws_task = asyncio.create_task(self.ws_server.start())
        gateway_task = asyncio.create_task(super().start())
        try:
            await asyncio.gather(ws_task, gateway_task)
        finally:
            for t in (
                ws_task,
                gateway_task,
                self._synth_worker_task,
                self._play_worker_task,
                self._watchdog_task,
                self._local_wakeup_task,
                self._mic_mute_task,
            ):
                if t is not None and not t.done():
                    t.cancel()

    async def _enter_idle(self) -> None:
        """Transition speaker to IDLE, reset pipelines, and stop recording when using local KWS."""
        self.state = SpeakerState.IDLE
        self.pipeline.reset_kws()
        self.pipeline.reset_vad()
        if self.config.local_kws and self.ws_server.connected:
            await self.ws_server.stop_recording()

    async def _on_speaker_connected(self) -> None:
        """Reset speaker audio pipeline state silently when the LX06 connects."""
        self.state = SpeakerState.IDLE
        self.pipeline.reset_kws()
        self.pipeline.reset_vad()
        logger.info(
            f"XiaoAi speaker ready. Say {self._all_wake_words} to wake up! "
            f"(active_role={self._active_role}, local_kws={self.config.local_kws}, "
            f"mic_muted={getattr(self, '_mic_muted', False)})"
        )

    async def _on_speaker_disconnected(self) -> None:
        """Cancel active playback when the speaker disconnects."""
        self._turn_in_progress = False
        await self._cancel_playback()
        self.state = SpeakerState.IDLE

    async def _on_speaker_event(self, event_name: str, data: Any) -> None:
        """Log speaker hardware/instruction events."""
        logger.debug(f"XiaoAi speaker event: {event_name} -> {data}")

    @staticmethod
    def _is_stop_keyword(keyword: str) -> bool:
        """Return True if a KWS trigger corresponds to a stop/interrupt command."""
        clean = keyword.strip()
        return any(sw == clean or sw in clean for sw in _KWS_STOP_WORDS)

    @staticmethod
    def _is_stop_utterance(text: str) -> bool:
        """Return True if an ASR utterance is a stop/interrupt command."""
        if text in _STOP_COMMANDS:
            return True
        return len(text) <= 6 and any(sub in text for sub in _CORE_STOP_SUBSTRINGS)

    @staticmethod
    def _is_clear_utterance(text: str) -> bool:
        """Return True if an ASR utterance requests starting a fresh conversation."""
        if text in _CLEAR_COMMANDS:
            return True
        return len(text) <= 8 and any(sub in text for sub in _CORE_CLEAR_SUBSTRINGS)

    async def _abort_active_turn(self) -> None:
        """Abort any active Kesoku turns across all speaker persona channels."""
        gateway = getattr(self, "gateway", None)
        db = getattr(gateway, "db", None) if gateway is not None else None
        abort_fn = getattr(gateway, "abort_session", None) if gateway is not None else None
        get_sess_fn = getattr(db, "get_session_by_channel", None) if db is not None else None
        if not callable(get_sess_fn) or not callable(abort_fn):
            return
        seen_session_ids: set[str] = set()
        for ch_id in self._all_channel_ids():
            try:
                session = await get_sess_fn(self.chatbot_id, ch_id)
                if session is None or session.id in seen_session_ids:
                    continue
                seen_session_ids.add(session.id)
                try:
                    await abort_fn(session.id, skip_memory_close=True)
                except TypeError:
                    await abort_fn(session.id)
            except Exception as e:
                logger.warning(f"Failed to abort active XiaoAi session on '{ch_id}': {e}")

    async def _interrupt_active_turns(self, *, suppress: bool = True) -> None:
        """Record superseded turn IDs, cancel queued/active playback, and abort backend workers."""
        self._record_aborted_turn(getattr(self, "_active_turn_id", None))
        self._record_aborted_turn(self._active_stream_turn_id)
        self._active_turn_id = None
        self._active_turn_channel_id = None
        self._active_stream_turn_id = None
        self._streamed_any_sentence = False
        self._turn_in_progress = False
        self._turn_start_ts = 0.0
        self._suppress_until_next_user_turn = suppress
        self._chunker.reset()
        await self._cancel_playback()
        await self._abort_active_turn()

    async def post_ingest_hook(self, session: Any, user_msg: Message, dto: InboundMessageDTO) -> None:
        """Bind `_active_turn_id` to the newly created inbound user message ID."""
        if getattr(self, "_turn_in_progress", False) and user_msg.channel_id == getattr(
            self, "_active_turn_channel_id", None
        ):
            self._active_turn_id = user_msg.id

    async def _mark_delivered(self, message: Message) -> None:
        """Mark `message` as DELIVERED in Kesoku DB when gateway DB is available."""
        gateway = getattr(self, "gateway", None)
        db = getattr(gateway, "db", None) if gateway is not None else None
        update_fn = getattr(db, "update_message_status", None) if db is not None else None
        if not callable(update_fn):
            return
        try:
            await update_fn(message.id, MessageStatus.DELIVERED)
        except Exception as e:
            logger.debug(f"Failed to mark message '{message.id}' as DELIVERED: {e}")

    async def _handle_stop_command(self, trigger: str) -> None:
        """Immediately stop audio output, abort active turns, and return speaker to IDLE."""
        logger.info(
            f"Stop command detected ('{trigger}', prev_state={self.state}) -> halting playback and active turn"
        )
        await self._interrupt_active_turns(suppress=True)
        await self._enter_idle()
        if self.ws_server.connected:
            await self.ws_server.play_pcm(self._sleep_chime)

    async def _on_local_wakeup(self, keyword: str) -> str:
        """Accept an on-speaker local KWS wakeup trigger without blocking the HTTP response."""
        if getattr(self, "_mic_muted", False):
            logger.debug(f"Ignoring local wakeup trigger ('{keyword}') while speaker mic is muted")
            return "muted"
        if self.state != SpeakerState.IDLE:
            logger.debug(
                f"Ignoring local wakeup trigger ('{keyword}') while speaker state={self.state}"
            )
            return "active"
        if self._local_wakeup_task is not None and not self._local_wakeup_task.done():
            return "active"
        self._local_wakeup_task = asyncio.create_task(self._activate_local_wakeup(keyword))
        return "ok"

    async def _on_mic_mute(self, muted: bool) -> str:
        """Accept a speaker hardware mic-mute state update without blocking the HTTP response."""
        self._mic_muted = muted
        self.ws_server.mic_muted = muted
        if getattr(self, "_mic_mute_task", None) is not None and not self._mic_mute_task.done():
            self._mic_mute_task.cancel()
        self._mic_mute_task = asyncio.create_task(self._sync_mic_mute_state())
        return "muted" if muted else "unmuted"

    async def _sync_mic_mute_state(self) -> None:
        """Apply the latest speaker hardware mic-mute state to recording and active turns."""
        try:
            async with self._audio_lock:
                if self._mic_muted:
                    logger.info("XiaoAi hardware microphone muted -> stopping recording and active turns")
                    if (
                        self.state != SpeakerState.IDLE
                        or getattr(self, "_turn_in_progress", False)
                        or getattr(self, "_active_turn_id", None) is not None
                    ):
                        await self._interrupt_active_turns(suppress=True)
                    self.state = SpeakerState.IDLE
                    self.pipeline.reset_kws()
                    self.pipeline.reset_vad()
                    if self.ws_server.connected:
                        await self.ws_server.stop_recording(force=True)
                else:
                    logger.info("XiaoAi hardware microphone unmuted -> resuming listening")
                    self.pipeline.reset_kws()
                    self.pipeline.reset_vad()
                    if not self.config.local_kws and self.ws_server.connected:
                        await self.ws_server.start_recording()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Failed to sync XiaoAi mic mute state (muted={self._mic_muted}): {e}")

    async def _activate_local_wakeup(self, keyword: str) -> None:
        """Start speaker microphone streaming and enter LISTENING after a local KWS trigger."""
        try:
            async with self._audio_lock:
                if (
                    self.state != SpeakerState.IDLE
                    or not self.ws_server.connected
                    or getattr(self, "_mic_muted", False)
                ):
                    return
                await self.ws_server.start_recording()
                await self._handle_wakeup(keyword)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Failed to activate local wakeup ('{keyword}'): {e}")

    async def _on_audio_frame(self, samples: np.ndarray) -> None:
        """Process each 16kHz float32 microphone frame through KWS / VAD / ASR."""
        if getattr(self, "_mic_muted", False):
            return
        async with self._audio_lock:
            if getattr(self, "_mic_muted", False):
                return
            state = self.state
            if state in (SpeakerState.IDLE, SpeakerState.SPEAKING, SpeakerState.THINKING):
                keyword = await asyncio.to_thread(self.pipeline.feed_kws, samples)
                if keyword:
                    if self._is_stop_keyword(keyword):
                        if state in (SpeakerState.SPEAKING, SpeakerState.THINKING):
                            await self._handle_stop_command(keyword)
                    else:
                        await self._handle_wakeup(keyword)
                return

            if state == SpeakerState.LISTENING:
                if time.monotonic() < self._vad_unmute_ts:
                    return
                is_speech, segments = await asyncio.to_thread(self.pipeline.feed_vad, samples)
                if is_speech:
                    self._listen_deadline = time.monotonic() + max(self.config.continuous_listen_sec, 2.5)
                for seg in segments:
                    await self._handle_utterance(seg)
                    break

    async def _handle_wakeup(self, keyword: str) -> None:
        """Handle custom wake word detection (including persona switching & barge-in)."""
        role = self._wake_word_to_role.get(keyword)
        if role is None:
            for ww, mapped_role in self._wake_word_to_role.items():
                if ww in keyword:
                    role = mapped_role
                    break
        if role is None:
            role = self._active_role

        logger.info(f"Wake word detected: '{keyword}' -> role='{role}' (prev_state={self.state})")
        if (
            self.state in (SpeakerState.SPEAKING, SpeakerState.THINKING)
            or getattr(self, "_turn_in_progress", False)
            or getattr(self, "_active_turn_id", None) is not None
        ):
            await self._interrupt_active_turns(suppress=True)

        if role != self._active_role:
            logger.info(f"Switching XiaoAi active persona: '{self._active_role}' -> '{role}'")
            self._active_role = role
        if isinstance(self.tts, MlxQwenTTSEngine):
            self.tts.set_active_voice(role)

        cue_bank = self._persona_cues.get(role)
        cue_pcm = cue_bank.pick("wakeup") if cue_bank is not None else None
        using_cue = cue_pcm is not None
        out_pcm = cue_pcm if cue_pcm is not None else self._wakeup_chime
        cue_dur = len(out_pcm) / (24000 * 2)
        logger.info(f"Playing wakeup {'cue' if using_cue else 'chime'} (role='{role}', duration={cue_dur:.2f}s)")
        self.pipeline.reset_vad()
        self._vad_unmute_ts = time.monotonic() + cue_dur + 0.15
        self.state = SpeakerState.LISTENING
        wake_window = self.config.continuous_listen_sec if self.config.continuous_listen_sec > 0 else 6.0
        self._listen_deadline = self._vad_unmute_ts + wake_window
        await self.ws_server.play_pcm(out_pcm)

    async def _handle_utterance(self, audio_f32: np.ndarray) -> None:
        """Transcribe a completed user speech segment and route to fast command or Kesoku."""
        text = await asyncio.to_thread(self.pipeline.transcribe, audio_f32)
        cleaned = text.strip(" ，。！？,.!?")
        if not cleaned:
            return

        # Strip leading wake word if the user spoke wake word + command in one breath
        persona_switched = False
        for ww in self._all_wake_words:
            if cleaned.startswith(ww):
                mapped_role = self._wake_word_to_role.get(ww)
                if mapped_role and mapped_role != self._active_role:
                    logger.info(
                        f"Switching XiaoAi active persona via utterance: '{self._active_role}' -> '{mapped_role}'"
                    )
                    persona_switched = True
                    self._active_role = mapped_role
                    if isinstance(self.tts, MlxQwenTTSEngine):
                        self.tts.set_active_voice(mapped_role)
                remainder = cleaned[len(ww) :].strip(" ，。！？,.!?")
                if not remainder:
                    return
                cleaned = remainder
                break

        logger.info(f"[XiaoAi ASR] User said (role={self._active_role}): {cleaned}")

        if await self._try_local_command(cleaned):
            return

        has_active_work = (
            persona_switched
            or getattr(self, "_turn_in_progress", False)
            or getattr(self, "_active_turn_id", None) is not None
            or getattr(self, "_synth_in_flight", False)
            or not self._tts_queue.empty()
            or not self._play_queue.empty()
        )
        if has_active_work:
            await self._interrupt_active_turns(suppress=False)

        self.state = SpeakerState.THINKING
        self._turn_start_ts = time.monotonic()
        self._last_tool_hint_ts = 0.0
        self._chunker.reset()
        self._active_stream_turn_id = None
        self._streamed_any_sentence = False

        channel_id = self._active_channel_id()
        custom_prompt = await self._compile_custom_prompt()
        await self._maybe_reset_idle_session(channel_id)
        dto = InboundMessageDTO(
            sender_id=SPEAKER_SENDER_ID,
            channel_id=channel_id,
            text=cleaned,
            message_id=str(uuid.uuid4()),
            session_title=f"智能音箱 ({self._active_role})",
            custom_prompt=custom_prompt,
            role=self._active_role,
        )
        self._active_turn_channel_id = channel_id
        self._active_turn_id = None
        self._suppress_until_next_user_turn = False
        self._turn_in_progress = True
        await self.ingest_message(dto)

    async def _maybe_reset_idle_session(self, channel_id: str | None = None) -> bool:
        """Start a fresh session if the current channel session has been idle past the threshold."""
        if self.config.session_idle_reset_sec <= 0:
            return False
        target_channel = channel_id or self._active_channel_id()
        session = await self.gateway.db.get_session_by_channel(self.chatbot_id, target_channel)
        if session is None:
            return False
        idle_sec = time.time() - float(session.updated_at)
        if idle_sec >= self.config.session_idle_reset_sec:
            logger.info(
                f"XiaoAi session '{session.id}' on '{target_channel}' idle for {idle_sec:.0f}s "
                f"(>= {self.config.session_idle_reset_sec:.0f}s); starting fresh session."
            )
            await self.clear_session_by_channel(target_channel)
            return True
        return False

    async def _try_local_command(self, text: str) -> bool:
        """Intercept instant local speaker commands without invoking the LLM."""
        if self._is_stop_utterance(text):
            await self._handle_stop_command(text)
            return True

        if self._is_clear_utterance(text):
            await self._interrupt_active_turns(suppress=False)
            await self.clear_session_by_channel(self._active_channel_id())
            await self._enqueue_tts("好的，已经开启新对话。")
            return True

        if text in _VOL_UP_COMMANDS:
            await self.ws_server.run_shell("mphelper volume_up >/dev/null 2>&1 || true")
            await self._enqueue_tts("音量调大了一点。")
            return True

        if text in _VOL_DOWN_COMMANDS:
            await self.ws_server.run_shell("mphelper volume_down >/dev/null 2>&1 || true")
            await self._enqueue_tts("音量调小了一点。")
            return True

        return False

    async def _compile_custom_prompt(self) -> str:
        """Build the speaker-specific system prompt addition."""
        default_prompt = (
            "# 智能音箱语音交互指南\n"
            "- 当前为纯语音播报场景，请用自然简洁的中文口语回复，默认控制在 1~3 句以内。\n"
            "- 不要输出 Markdown 格式符号、代码块、表格、URL 或 `[question]` / `ui` 块。\n"
            "- 如需列举内容，请用口语自然串联，不要使用编号清单。\n"
        )
        if self.config.sys_prompt_file:
            sys_file = self.config.sys_prompt_file
            cfg = get_config()
            if not os.path.isabs(sys_file) and cfg.agent_working_dir:
                sys_file = os.path.join(cfg.agent_working_dir, sys_file)
            if await async_exists(sys_file):
                try:
                    content = await async_read_text_file(sys_file)
                    if content.strip():
                        return content.strip()
                except Exception as e:
                    logger.warning(f"Failed to read XiaoAi sys_prompt_file '{sys_file}': {e}")
        return default_prompt

    async def _handle_stream_update(
        self,
        channel_id: str,
        session_id: str,
        turn_id: str,
        text: str | None,
    ) -> None:
        """Receive 50ms-throttled LLM stream deltas and enqueue complete sentences for TTS."""
        if (
            not self.ws_server.connected
            or getattr(self, "_suppress_until_next_user_turn", False)
            or self._is_stale_turn(channel_id=channel_id, turn_id=turn_id)
        ):
            return
        if text is None:
            self._chunker.reset()
            self._active_stream_turn_id = None
            return

        if self._active_stream_turn_id != turn_id:
            self._active_stream_turn_id = turn_id
            self._streamed_any_sentence = False
            self._chunker.reset()

        sentences = self._chunker.feed_accumulated(text)
        for sentence in sentences:
            self._streamed_any_sentence = True
            await self._enqueue_tts(sentence)

    def supports_intermediate_messages(self) -> bool:
        """Enable intermediate hooks for instant tool-call transition cues when configured."""
        return self.config.progress_feedback == "status"

    async def handle_intermediate_message(self, message: Message) -> None:
        """Speak a brief cached transition cue immediately when a tool call starts."""
        try:
            if (
                self.config.progress_feedback != "status"
                or not self.ws_server.connected
                or getattr(self, "_suppress_until_next_user_turn", False)
                or self._is_stale_turn(
                    channel_id=getattr(message, "channel_id", None),
                    turn_id=getattr(message, "parent_id", None),
                )
            ):
                return
            if message.role != MessageRole.TOOL or message.type != MessageType.TOOL_CALL:
                return
            now = time.monotonic()
            if now - self._last_tool_hint_ts < 6.0:
                return
            self._last_tool_hint_ts = now
            meta = message.metadata if isinstance(message.metadata, dict) else {}
            tool_name = str(meta.get("tool_name") or message.sender or "")
            category = "tool_search" if tool_name in _SEARCH_TOOLS else "tool_work"
            cue_bank = self._persona_cues.get(self._active_role)
            cue_pcm = cue_bank.pick(category) if cue_bank is not None else None
            logger.info(
                f"Playing tool cue (role='{self._active_role}', category='{category}', "
                f"tool='{tool_name}', cached={cue_pcm is not None})"
            )
            if cue_pcm is not None:
                await self._tts_queue.put((self._tts_generation, self._active_role, "", cue_pcm))
                return
            hint = _TOOL_VOICE_HINTS.get(tool_name, _DEFAULT_TOOL_VOICE_HINT)
            await self._enqueue_tts(hint)
        finally:
            await self._mark_delivered(message)

    async def handle_message(self, message: Message) -> None:
        """Render outbound assistant messages via the standard Chatbot segment pipeline."""
        is_tool_result = message.role == MessageRole.TOOL and message.type != MessageType.TOOL_CALL
        is_final = not self.is_intermediate_message(message) and not is_tool_result
        channel_id = getattr(message, "channel_id", None)
        turn_id = getattr(message, "parent_id", None)
        stale = self._is_stale_turn(channel_id=channel_id, turn_id=turn_id)
        suppressed = is_final and getattr(self, "_suppress_until_next_user_turn", False)
        if stale or suppressed:
            logger.debug(
                f"Dropping {'stale' if stale else 'suppressed'} XiaoAi message "
                f"(turn_id={turn_id}, channel={channel_id})"
            )
            if suppressed and not stale:
                self._turn_in_progress = False
            await self._mark_delivered(message)
            return
        try:
            await self.render_outgoing_message(message)
        finally:
            if is_final and not self._is_stale_turn(channel_id=channel_id, turn_id=turn_id):
                self._turn_in_progress = False
                if turn_id and getattr(self, "_active_turn_id", None) == turn_id:
                    self._active_turn_id = None

    async def _enqueue_error_cue(self) -> bool:
        """Play a cached persona error cue if available, returning True when queued."""
        if not self.ws_server.connected:
            return False
        cue_bank = self._persona_cues.get(self._active_role)
        cue_pcm = cue_bank.pick("error") if cue_bank is not None else None
        if cue_pcm is None:
            return False
        logger.info(f"Playing error cue (role='{self._active_role}')")
        await self._tts_queue.put((self._tts_generation, self._active_role, "", cue_pcm))
        return True

    async def send_text_chunks(self, channel_id: str, chunks: list[str], message: Message) -> None:
        """Flush remaining streamed text or speak full text for non-streamed messages."""
        if not self.ws_server.connected:
            return
        full_text = "".join(chunks).strip()
        if not full_text:
            return
        if full_text.startswith("⚠️") and not self._streamed_any_sentence:
            if await self._enqueue_error_cue():
                self._active_stream_turn_id = None
                return

        if self._streamed_any_sentence:
            tail_sentences = self._chunker.flush(full_text)
            for s in tail_sentences:
                await self._enqueue_tts(s)
        else:
            self._chunker.reset()
            for s in self._chunker.flush(full_text):
                await self._enqueue_tts(s)

        self._active_stream_turn_id = None
        self._streamed_any_sentence = False

    async def send_voice_segment(self, channel_id: str, file_path: str, message: Message) -> None:
        """Play a `[voice: path]` audio file directly on the LX06 speaker."""
        if not self.ws_server.connected:
            return
        try:
            pcm = await decode_audio_file_to_pcm24k(file_path)
            if pcm:
                await self._tts_queue.put((self._tts_generation, self._active_role, "", pcm))
        except Exception as e:
            logger.warning(f"Failed to play voice segment '{file_path}' on XiaoAi: {e}")

    async def send_file_segment(self, channel_id: str, file_path: str, message: Message) -> None:
        """Ignore visual file attachments on an audio-only smart speaker."""
        logger.info(f"XiaoAi speaker skipping non-audio file attachment: {file_path}")

    async def _enqueue_tts(self, text: str) -> None:
        """Push a sentence onto the async TTS synthesis/playback queue."""
        if text.strip():
            await self._tts_queue.put((self._tts_generation, self._active_role, text.strip(), None))

    @staticmethod
    def _drain_queue(queue: asyncio.Queue[Any]) -> None:
        """Remove all pending items from an asyncio queue and mark them done."""
        while not queue.empty():
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def _cancel_playback(self) -> None:
        """Immediately halt audio playback and drop all queued TTS sentences and audio chunks."""
        self._tts_generation += 1
        self._drain_queue(self._tts_queue)
        self._drain_queue(self._play_queue)
        self._play_end_ts = 0.0
        await self.ws_server.stop_play()

    async def _sleep_interruptible(self, duration: float, gen: int) -> None:
        """Sleep up to `duration` seconds, waking early on cancellation or disconnect."""
        deadline = time.monotonic() + max(0.0, duration)
        while True:
            if gen != self._tts_generation or not self.ws_server.connected:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.1, remaining))

    async def _synth_worker(self) -> None:
        """Synthesize queued sentences into PCM chunks ahead of playback."""
        while True:
            gen, voice_id, text, pre_pcm = await self._tts_queue.get()
            self._synth_in_flight = True
            try:
                if gen != self._tts_generation or not self.ws_server.connected:
                    continue
                if pre_pcm is not None:
                    await self._play_queue.put((gen, pre_pcm))
                    continue

                if isinstance(self.tts, MlxQwenTTSEngine):
                    async for pcm_chunk in self.tts.synthesize_stream(
                        text,
                        voice_id=voice_id,
                        is_cancelled=lambda g=gen: g != self._tts_generation or not self.ws_server.connected,
                    ):
                        if gen != self._tts_generation or not self.ws_server.connected:
                            break
                        if pcm_chunk:
                            await self._play_queue.put((gen, pcm_chunk))
                else:
                    synth_stream = getattr(self.tts, "synthesize_stream", None)
                    if callable(synth_stream):
                        async for pcm_chunk in synth_stream(
                            text,
                            is_cancelled=lambda g=gen: g != self._tts_generation or not self.ws_server.connected,
                        ):
                            if gen != self._tts_generation or not self.ws_server.connected:
                                break
                            if pcm_chunk:
                                await self._play_queue.put((gen, pcm_chunk))
                    else:
                        pcm = await self.tts.synthesize(text)
                        if gen == self._tts_generation and self.ws_server.connected and pcm:
                            await self._play_queue.put((gen, pcm))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"XiaoAi TTS synthesis error for '{text[:30]}': {e}")
                if gen == self._tts_generation and self.ws_server.connected:
                    cue_bank = self._persona_cues.get(voice_id) or self._persona_cues.get(self._active_role)
                    cue_pcm = cue_bank.pick("error") if cue_bank is not None else None
                    if cue_pcm is not None:
                        await self._play_queue.put((gen, cue_pcm))
            finally:
                self._synth_in_flight = False
                self._tts_queue.task_done()

    async def _finish_speaking_turn(self) -> None:
        """Transition out of SPEAKING once all queued playback has drained."""
        if self.state != SpeakerState.SPEAKING:
            return
        if self.config.continuous_listen_sec > 0:
            self.pipeline.reset_vad()
            self.state = SpeakerState.LISTENING
            self._listen_deadline = time.monotonic() + self.config.continuous_listen_sec
        else:
            await self._enter_idle()

    async def _play_worker(self) -> None:
        """Stream synthesized PCM chunks continuously to the LX06 speaker."""
        while True:
            if self.state == SpeakerState.SPEAKING:
                try:
                    gen, pcm = await asyncio.wait_for(self._play_queue.get(), timeout=0.15)
                except TimeoutError:
                    if (
                        self._play_queue.empty()
                        and self._tts_queue.empty()
                        and not self._synth_in_flight
                        and time.monotonic() >= self._play_end_ts
                    ):
                        if getattr(self, "_turn_in_progress", False):
                            self.state = SpeakerState.THINKING
                        else:
                            await self._finish_speaking_turn()
                    continue
            else:
                gen, pcm = await self._play_queue.get()

            try:
                if gen != self._tts_generation or not self.ws_server.connected or not pcm:
                    continue
                self.state = SpeakerState.SPEAKING

                ahead = self._play_end_ts - time.monotonic()
                if ahead > 0.8:
                    await self._sleep_interruptible(ahead - 0.8, gen)
                    if gen != self._tts_generation or not self.ws_server.connected:
                        continue

                await self.ws_server.play_pcm(pcm)
                duration_sec = len(pcm) / (24000 * 2)
                self._play_end_ts = max(time.monotonic(), self._play_end_ts) + duration_sec
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"XiaoAi speaker playback error: {e}")
            finally:
                self._play_queue.task_done()

    async def _listen_watchdog(self) -> None:
        """Return to IDLE when continuous listening expires or recover a stalled THINKING turn."""
        while True:
            await asyncio.sleep(0.5)
            now = time.monotonic()
            if self.state == SpeakerState.LISTENING and now > self._listen_deadline:
                logger.info("Continuous listening window expired -> returning to IDLE")
                await self._enter_idle()
                if self.ws_server.connected:
                    await self.ws_server.play_pcm(self._sleep_chime)
            elif (
                self.state == SpeakerState.THINKING
                and self._turn_in_progress
                and self._turn_start_ts > 0.0
                and (now - self._turn_start_ts) > 60.0
            ):
                logger.warning("XiaoAi turn timed out in THINKING (>60s); aborting turn and playing error fallback")
                await self._interrupt_active_turns(suppress=True)
                if not await self._enqueue_error_cue():
                    await self._enter_idle()
