"""Unit tests for the Open-XiaoAI smart speaker adapter components."""

import numpy as np

from kesoku.config import KesokuConfig, XiaoAiConfig
from kesoku.gateway.chatbot.xiaoai.audio_pipeline import chinese_phrase_to_kws_tokens
from kesoku.gateway.chatbot.xiaoai.tts import (
    MlxQwenTTSEngine,
    SentenceChunker,
    VoiceCueBank,
    clean_text_for_speech,
    detect_tts_lang_code,
    float32_to_pcm16,
    make_wakeup_chime,
    normalize_pcm16_rms,
    split_tts_language_segments,
    trim_pcm16_silence,
)
from kesoku.gateway.chatbot.xiaoai.ws_server import OpenXiaoAiServer, decode_a113_s32_to_s16


def test_decode_a113_s32_to_s16():
    # A113 PDM stores 24-bit audio in the lower 24 bits of S32_LE -> right-shift by 8
    raw_samples = np.array([0, 256, -256, 32767 * 256, -32768 * 256, 100000 * 256], dtype="<i4")
    out = decode_a113_s32_to_s16(raw_samples.tobytes())
    assert out.dtype == np.int16
    assert out.tolist() == [0, 1, -1, 32767, -32768, 32767]


def test_clean_text_for_speech():
    raw = "**你好**，看这个 [链接](https://example.com) 和 `code` ```python\nprint(1)\n``` [voice: a.wav]"
    cleaned = clean_text_for_speech(raw)
    assert "**" not in cleaned
    assert "https://" not in cleaned
    assert "[voice:" not in cleaned
    assert "你好" in cleaned
    assert "链接" in cleaned
    assert "代码内容请在屏幕查看" in cleaned


def test_sentence_chunker_streaming():
    chunker = SentenceChunker(soft_min_chars=12)
    assert chunker.feed_accumulated("今天天气") == []
    s1 = chunker.feed_accumulated("今天天气真不错。明天")
    assert s1 == ["今天天气真不错。"]
    s2 = chunker.feed_accumulated("今天天气真不错。明天可能会下雨，记得带伞！")
    assert len(s2) >= 1
    tail = chunker.flush("今天天气真不错。明天可能会下雨，记得带伞！晚安")
    assert tail == ["晚安"]


def test_chinese_phrase_to_kws_tokens():
    line = chinese_phrase_to_kws_tokens("小张小张")
    assert line.endswith("@小张小张")
    assert "x" in line and "zh" in line
    # Pass-through raw token format
    raw = "x iǎo zh āng @小张"
    assert chinese_phrase_to_kws_tokens(raw) == raw
    # Per-keyword score and threshold overrides
    stop_line = chinese_phrase_to_kws_tokens("闭嘴", score=1.5, threshold=0.35)
    assert stop_line == "b ì z uǐ :1.5 #0.35 @闭嘴"


def test_normalize_pcm16_rms():
    quiet = np.full(1200, 1000, dtype="<i2").tobytes()  # rms ~= 0.0305
    boosted = normalize_pcm16_rms(quiet, target_rms=0.11, max_peak=0.85, max_gain=3.5)
    samples = np.frombuffer(boosted, dtype="<i2")
    assert samples.size == 1200
    assert int(np.max(np.abs(samples))) > 3000
    assert int(np.max(np.abs(samples))) <= int(0.85 * 32767) + 1
    # Clip with peak near max_peak is capped smoothly without clipping
    spiky = np.array([20000] + [100] * 1199, dtype="<i2").tobytes()
    capped = normalize_pcm16_rms(spiky, target_rms=0.2, max_peak=0.85, max_gain=4.0)
    assert int(np.max(np.abs(np.frombuffer(capped, dtype="<i2")))) <= int(0.85 * 32767) + 1


def test_make_wakeup_chime():
    pcm = make_wakeup_chime()
    assert len(pcm) > 0
    assert len(pcm) % 2 == 0


def test_xiaoai_config_in_root_config():
    cfg = KesokuConfig(
        chatbots={
            "xiaoai": [
                XiaoAiConfig(
                    enabled=True,
                    chatbot_id="xiaoai_home",
                    default_role="xiaozhang",
                    wake_words=["小张小张"],
                )
            ]
        }
    )
    assert len(cfg.active_xiaoais) == 1
    assert cfg.get_xiaoai_config("xiaoai_home") is not None
    assert cfg.get_chatbot_default_role("xiaoai_home") == "xiaozhang"


def test_float32_to_pcm16():
    arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    pcm = float32_to_pcm16(arr)
    samples = np.frombuffer(pcm, dtype="<i2").tolist()
    assert samples == [-32767, -32767, 0, 32767, 32767]
    assert float32_to_pcm16(np.empty(0, dtype=np.float32)) == b""


def test_xiaoai_config_qwen_tts():
    cfg = XiaoAiConfig(
        enabled=True,
        tts_provider="qwen_tts",
        qwen_tts_model="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        qwen_tts_ref_audio="roles/xiaozhang/audio/sample.wav",
    )
    assert cfg.tts_provider == "qwen_tts"
    assert cfg.qwen_tts_model == "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
    assert cfg.qwen_tts_ref_audio == "roles/xiaozhang/audio/sample.wav"


async def test_mlx_qwen_tts_ref_text_and_cache():
    engine = MlxQwenTTSEngine(
        model_id="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        ref_audio="/tmp/今天有不少好朋友呢到了现场助阵.wav",
    )
    assert engine.ref_text == "今天有不少好朋友呢到了现场助阵"

    calls = 0

    def fake_synth(text: str) -> bytes:
        nonlocal calls
        calls += 1
        return b"\x01\x00" * 100

    engine._synthesize_sync = fake_synth  # type: ignore[method-assign]
    out1 = await engine.synthesize("你好")
    out2 = await engine.synthesize("你好")
    assert out1 == out2
    assert len(out1) == 200
    assert calls == 1
    assert await engine.synthesize("   ") == b""


async def test_mlx_qwen_tts_stream_and_cancel():
    import time
    from types import SimpleNamespace

    class FakeModel:
        sample_rate = 24000

        def __init__(self) -> None:
            self.yielded = 0

        def generate(self, **kwargs):
            assert kwargs.get("stream") is True
            for i in range(4):
                time.sleep(0.015)
                self.yielded += 1
                yield SimpleNamespace(audio=np.full(240, 0.25 * (i + 1), dtype=np.float32), sample_rate=24000)

    engine = MlxQwenTTSEngine(
        model_id="fake-model",
        ref_audio="/tmp/今天有不少好朋友呢到了现场助阵.wav",
    )
    fake_model = FakeModel()
    engine._model = fake_model
    engine._ref_audio_mx = np.zeros(240, dtype=np.float32)

    chunks = [chunk async for chunk in engine.synthesize_stream("你好世界")]
    assert len(chunks) == 4
    assert all(len(c) == 480 for c in chunks)
    # Cached short phrase returns combined bytes in 1 chunk without calling model again
    cached_chunks = [chunk async for chunk in engine.synthesize_stream("你好世界")]
    assert cached_chunks == [b"".join(chunks)]
    assert fake_model.yielded == 4

    # Cancellation stops generation mid-stream and does not cache partial output
    cancel_flag = False

    def is_cancelled() -> bool:
        return cancel_flag

    partial_chunks = []
    async for chunk in engine.synthesize_stream("另一句很长的话不要缓存", is_cancelled=is_cancelled):
        partial_chunks.append(chunk)
        if len(partial_chunks) == 2:
            cancel_flag = True

    assert len(partial_chunks) == 2
    assert fake_model.yielded < 8
    assert "另一句很长的话不要缓存" not in engine._cache


def test_trim_pcm16_silence():
    silence = np.zeros(1000, dtype="<i2")
    speech = np.full(200, 2000, dtype="<i2")
    raw = np.concatenate([silence, speech, silence]).tobytes()
    trimmed = trim_pcm16_silence(raw, threshold=300, pad_samples=50)
    samples = np.frombuffer(trimmed, dtype="<i2")
    assert len(samples) == 300
    assert np.all(samples[50:250] == 2000)
    assert trim_pcm16_silence(silence.tobytes(), threshold=300) == b""


async def test_voice_cue_bank_load_and_pick(tmp_path):
    import wave

    wakeup_dir = tmp_path / "wakeup"
    wakeup_dir.mkdir()
    for name, amp, length in [("a.wav", 1500, 1200), ("b.wav", 2500, 1600)]:
        p = wakeup_dir / name
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(np.full(length, amp, dtype="<i2").tobytes())

    bank = VoiceCueBank()
    total = await bank.load_dir(tmp_path)
    assert total == 2
    assert bank.counts() == {"wakeup": 2}
    assert bank.pick("missing") is None
    p1 = bank.pick("wakeup")
    p2 = bank.pick("wakeup")
    assert p1 is not None and p2 is not None
    assert p1 != p2

    cfg = XiaoAiConfig(enabled=True, cues_dir="roles/leijun/cues")
    assert cfg.cues_dir == "roles/leijun/cues"


async def test_session_idle_reset():
    import time
    from types import SimpleNamespace

    from kesoku.gateway.chatbot.xiaoai.adapter import XiaoAiChatbot

    assert XiaoAiConfig().session_idle_reset_sec == 1800.0
    assert XiaoAiConfig(session_idle_reset_sec=0.0).session_idle_reset_sec == 0.0

    cleared: list[str] = []

    async def fake_clear(channel_id: str) -> str:
        cleared.append(channel_id)
        return "ok"

    current_session = SimpleNamespace(id="sess1", updated_at=time.time() - 2000.0)

    async def fake_get_session(chatbot_id: str, channel_id: str):
        return current_session

    bot = XiaoAiChatbot.__new__(XiaoAiChatbot)
    bot.chatbot_id = "xiaoai"
    bot.config = XiaoAiConfig(session_idle_reset_sec=1800.0)
    bot.gateway = SimpleNamespace(db=SimpleNamespace(get_session_by_channel=fake_get_session))
    bot.clear_session_by_channel = fake_clear  # type: ignore[method-assign]

    # Idle >= 1800s -> resets session
    assert await bot._maybe_reset_idle_session() is True
    assert cleared == ["xiaoai:default"]

    # Active recently -> keeps session
    cleared.clear()
    current_session.updated_at = time.time() - 60.0
    assert await bot._maybe_reset_idle_session() is False
    assert cleared == []

    # Disabled (0) -> never resets
    current_session.updated_at = time.time() - 99999.0
    bot.config = XiaoAiConfig(session_idle_reset_sec=0.0)
    assert await bot._maybe_reset_idle_session() is False
    assert cleared == []


def test_xiaoai_multi_persona_config():
    cfg = XiaoAiConfig(
        enabled=True,
        default_role="leijun",
        personas=[
            {
                "role": "leijun",
                "wake_words": ["小雷小雷"],
                "qwen_tts_ref_audio": "roles/leijun/audio/a.wav",
                "cues_dir": "roles/leijun/cues",
            },
            {
                "role": "xiaozhang",
                "wake_words": ["小张小张"],
                "qwen_tts_ref_audio": "roles/xiaozhang/audio/b.wav",
                "cues_dir": "roles/xiaozhang/cues",
            },
        ],
    )
    assert len(cfg.personas) == 2
    assert cfg.personas[0].role == "leijun"
    assert cfg.personas[1].wake_words == ["小张小张"]


async def test_mlx_qwen_tts_multi_voice_and_cache_isolation():
    from types import SimpleNamespace

    seen_ref_texts: list[str] = []

    class FakeMultiModel:
        sample_rate = 24000

        def generate(self, **kwargs):
            seen_ref_texts.append(kwargs["ref_text"])
            val = 0.2 if kwargs["ref_text"] == "雷军参考" else 0.5
            yield SimpleNamespace(audio=np.full(240, val, dtype=np.float32), sample_rate=24000)

    engine = MlxQwenTTSEngine(model_id="fake-model")
    engine.register_voice("leijun", "/tmp/雷军参考.wav")
    engine.register_voice("xiaozhang", "/tmp/小张参考.wav")
    engine._model = FakeMultiModel()
    engine._voice_arrays["leijun"] = (np.zeros(240, dtype=np.float32), "雷军参考")
    engine._voice_arrays["xiaozhang"] = (np.zeros(240, dtype=np.float32), "小张参考")

    pcm_lei = await engine.synthesize("你好", voice_id="leijun")
    pcm_zhang = await engine.synthesize("你好", voice_id="xiaozhang")
    assert pcm_lei != pcm_zhang
    assert seen_ref_texts == ["雷军参考", "小张参考"]

    # Subsequent calls hit separate per-voice cache keys
    assert await engine.synthesize("你好", voice_id="leijun") == pcm_lei
    assert await engine.synthesize("你好", voice_id="xiaozhang") == pcm_zhang
    assert len(seen_ref_texts) == 2

    engine.set_active_voice("xiaozhang")
    assert engine.ref_text == "小张参考"


def _make_stub_xiaoai_bot(*, progress_feedback: str = "status"):
    import asyncio
    from types import SimpleNamespace

    from kesoku.constants import MessageStatus
    from kesoku.gateway.chatbot.xiaoai.adapter import SpeakerState, XiaoAiChatbot
    from kesoku.gateway.chatbot.xiaoai.tts import make_sleep_chime

    aborted: list[str] = []
    delivered: list[tuple[str, MessageStatus]] = []
    stopped_play = 0
    played_pcms: list[bytes] = []
    kws_resets = 0
    vad_resets = 0

    async def fake_get_session(chatbot_id: str, channel_id: str):
        return SimpleNamespace(id="sess_active")

    async def fake_abort(session_id: str) -> None:
        aborted.append(session_id)

    async def fake_update_status(msg_id: str, status: MessageStatus) -> None:
        delivered.append((msg_id, status))

    async def fake_stop_play() -> None:
        nonlocal stopped_play
        stopped_play += 1

    async def fake_play_pcm(pcm: bytes) -> None:
        played_pcms.append(pcm)

    def fake_reset_kws() -> None:
        nonlocal kws_resets
        kws_resets += 1

    def fake_reset_vad() -> None:
        nonlocal vad_resets
        vad_resets += 1

    bot = XiaoAiChatbot.__new__(XiaoAiChatbot)
    bot.chatbot_id = "xiaoai"
    bot.config = XiaoAiConfig(
        enabled=True,
        progress_feedback=progress_feedback,
        continuous_listen_sec=8.0,
    )
    bot.state = SpeakerState.IDLE
    bot._personas = {"leijun": object()}
    bot._active_role = "leijun"
    bot._persona_cues = {"leijun": VoiceCueBank()}
    bot._tts_queue = asyncio.Queue()
    bot._play_queue = asyncio.Queue(maxsize=32)
    bot._tts_generation = 0
    bot._synth_in_flight = False
    bot._play_end_ts = 0.0
    bot._last_tool_hint_ts = 0.0
    bot._listen_deadline = 0.0
    bot._turn_start_ts = 0.0
    bot._vad_unmute_ts = 0.0
    bot._turn_in_progress = False
    bot._suppress_until_next_user_turn = False
    bot._active_turn_id = None
    bot._active_turn_channel_id = None
    bot._aborted_turn_ids = set()
    bot._active_stream_turn_id = None
    bot._streamed_any_sentence = False
    bot._mic_muted = False
    bot._mic_mute_task = None
    bot._active_led = None
    bot._chunker = SentenceChunker()
    bot._sleep_chime = make_sleep_chime()
    bot._wakeup_chime = make_wakeup_chime()
    bot._audio_lock = asyncio.Lock()
    bot.ws_server = SimpleNamespace(connected=True, stop_play=fake_stop_play, play_pcm=fake_play_pcm)
    bot.pipeline = SimpleNamespace(
        feed_kws=lambda s: None,
        reset_kws=fake_reset_kws,
        reset_vad=fake_reset_vad,
    )
    bot.gateway = SimpleNamespace(
        db=SimpleNamespace(
            get_session_by_channel=fake_get_session,
            update_message_status=fake_update_status,
        ),
        abort_session=fake_abort,
    )
    bot._test_state = SimpleNamespace(
        aborted=aborted,
        delivered=delivered,
        get_stopped_play=lambda: stopped_play,
        played_pcms=played_pcms,
        get_kws_resets=lambda: kws_resets,
        get_vad_resets=lambda: vad_resets,
    )
    return bot


async def test_tool_call_intermediate_cue_dispatch():
    from kesoku.constants import MessageRole, MessageStatus, MessageType
    from kesoku.db import Message

    bot = _make_stub_xiaoai_bot(progress_feedback="status")
    assert bot.supports_intermediate_messages() is True
    assert XiaoAiConfig(progress_feedback="off").progress_feedback == "off"

    cue_bytes = b"\x11\x22" * 480
    bot._persona_cues["leijun"]._clips["tool_search"] = [cue_bytes]

    msg = Message(
        id="tc_1",
        session_id="sess_active",
        chatbot_id="xiaoai",
        channel_id="xiaoai:default",
        sender="Kesoku",
        role=MessageRole.TOOL,
        type=MessageType.TOOL_CALL,
        content="Calling tool `web_search`",
        metadata={"tool_name": "web_search"},
    )
    await bot.handle_message(msg)

    assert bot._tts_queue.qsize() == 1
    gen, voice, text, pre_pcm = bot._tts_queue.get_nowait()
    assert (gen, voice, text, pre_pcm) == (0, "leijun", "", cue_bytes)
    assert bot._test_state.delivered == [("tc_1", MessageStatus.DELIVERED)]


async def test_stop_command_kws_and_fuzzy_utterance():
    from kesoku.gateway.chatbot.xiaoai.adapter import SpeakerState

    bot = _make_stub_xiaoai_bot()

    # Stop keyword ignored while already IDLE
    bot.state = SpeakerState.IDLE
    bot.pipeline.feed_kws = lambda s: "闭嘴"
    await bot._on_audio_frame(np.ones(512, dtype=np.float32))
    assert bot._test_state.aborted == []
    assert bot._test_state.played_pcms == []

    # Stop keyword during SPEAKING cancels playback, aborts session, suppresses late stream, returns IDLE
    bot.state = SpeakerState.SPEAKING
    bot._turn_in_progress = True
    await bot._tts_queue.put((0, "leijun", "没说完", None))
    await bot._play_queue.put((0, b"\x01\x02" * 100))
    await bot._on_audio_frame(np.ones(512, dtype=np.float32))

    assert bot.state == SpeakerState.IDLE
    assert bot._turn_in_progress is False
    assert bot._suppress_until_next_user_turn is True
    assert bot._tts_queue.empty()
    assert bot._play_queue.empty()
    assert bot._tts_generation == 1
    assert bot._test_state.get_stopped_play() == 1
    assert bot._test_state.aborted == ["sess_active"]
    assert bot._test_state.played_pcms == [bot._sleep_chime]

    # Late streamed text from cancelled turn is suppressed
    await bot._handle_stream_update("xiaoai:default", "sess_active", "turn_1", "迟到的恢复句子。")
    assert bot._tts_queue.empty()

    # Fuzzy short stop utterances match, while normal long sentences containing '闭嘴' do NOT match
    assert await bot._try_local_command("闭嘴吧") is True
    assert await bot._try_local_command("停一下") is True
    assert await bot._try_local_command("帮我查一下闭嘴是什么意思") is False


async def test_tool_cue_returns_to_thinking_while_turn_in_progress():
    import asyncio

    from kesoku.gateway.chatbot.xiaoai.adapter import SpeakerState

    bot = _make_stub_xiaoai_bot()
    bot.state = SpeakerState.SPEAKING
    bot._turn_in_progress = True
    bot._play_end_ts = 0.0

    worker = asyncio.create_task(bot._play_worker())
    await asyncio.sleep(0.25)
    assert bot.state == SpeakerState.THINKING

    # Once final response chunks play and finish (_turn_in_progress=False), draining transitions to LISTENING
    bot._turn_in_progress = False
    await bot._play_queue.put((0, b"\x00\x00" * 240))
    await asyncio.sleep(0.25)
    assert bot.state == SpeakerState.LISTENING
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass


async def test_ws_server_play_epoch_cancels_inflight_play():
    import asyncio

    sent_frames: list[bytes | str] = []

    class FakeWS:
        async def send(self, data: bytes | str) -> None:
            sent_frames.append(data)
            await asyncio.sleep(0.01)

    server = OpenXiaoAiServer()
    server._ws = FakeWS()
    server._play_active = True

    big_pcm = b"\x01\x00" * (2400 * 10)  # 10 chunks
    play_task = asyncio.create_task(server.play_pcm(big_pcm))
    await asyncio.sleep(0.025)
    await server.stop_play()
    # Immediately reactivate playback as a new cue/wakeup would
    await server.start_play()
    await play_task

    play_packets = [f for f in sent_frames if isinstance(f, bytes)]
    assert 1 <= len(play_packets) < 10
    assert server._play_epoch == 1


def test_detect_and_split_tts_language_segments():
    assert detect_tts_lang_code("今天天气真不错。") == "auto"
    assert detect_tts_lang_code("今日はいい天気です。") == "japanese"

    assert split_tts_language_segments("今天天气真不错。") == [("今天天气真不错。", "auto")]
    assert split_tts_language_segments("Are you OK？今天很棒。") == [("Are you OK？今天很棒。", "auto")]
    assert split_tts_language_segments("今日はいい天気ですね。") == [("今日はいい天気ですね。", "japanese")]
    assert split_tts_language_segments("今日は「東京」に行きます。") == [("今日は「東京」に行きます。", "japanese")]
    assert split_tts_language_segments("去居酒屋可以说「すみません、生ビールを一つください」。") == [
        ("去居酒屋可以说", "auto"),
        ("すみません、生ビールを一つください", "japanese"),
    ]
    assert split_tts_language_segments("日语里「ありがとう」就是谢谢。") == [
        ("日语里", "auto"),
        ("ありがとう", "japanese"),
        ("就是谢谢。", "auto"),
    ]
    assert split_tts_language_segments("这句话用日语说是：おはようございます！") == [
        ("这句话用日语说是", "auto"),
        ("おはようございます！", "japanese"),
    ]
    assert split_tts_language_segments("私の名前は雷軍です，よろしく。") == [
        ("私の名前は雷軍です，よろしく。", "japanese"),
    ]


async def test_mlx_qwen_tts_passes_japanese_lang_code():
    from types import SimpleNamespace

    calls: list[tuple[str, str, bool]] = []

    class FakeBilingualModel:
        sample_rate = 24000

        def generate(self, **kwargs):
            calls.append((kwargs["text"], kwargs["lang_code"], bool(kwargs.get("stream", False))))
            yield SimpleNamespace(audio=np.full(240, 0.2, dtype=np.float32), sample_rate=24000)

    engine = MlxQwenTTSEngine(model_id="fake-model")
    engine.register_voice("leijun", "/tmp/雷军参考.wav")
    engine._model = FakeBilingualModel()
    engine._voice_arrays["leijun"] = (np.zeros(240, dtype=np.float32), "雷军参考")

    pcm_sync = await engine.synthesize("可以说「おはようございます」。", voice_id="leijun")
    assert len(pcm_sync) == 240 * 2 * 2
    assert calls == [
        ("可以说", "auto", False),
        ("おはようございます", "japanese", False),
    ]

    calls.clear()
    chunks = [chunk async for chunk in engine.synthesize_stream("打招呼说「こんにちは」。", voice_id="leijun")]
    assert len(chunks) == 2
    assert calls == [
        ("打招呼说", "auto", True),
        ("こんにちは", "japanese", True),
    ]


async def test_error_cue_on_warning_message_and_tts_failure():
    import asyncio
    from types import SimpleNamespace

    bot = _make_stub_xiaoai_bot()
    err_pcm = b"\x05\x00" * 480
    bot._persona_cues["leijun"]._clips["error"] = [err_pcm]

    # 1. Warning error message from Kesoku plays cached error cue instead of raw English text
    await bot.send_text_chunks(
        "xiaoai:default",
        ["⚠️ The model returned an empty response after 3 attempts"],
        SimpleNamespace(id="m_warn"),
    )
    assert bot._tts_queue.qsize() == 1
    gen, voice, text, pre_pcm = bot._tts_queue.get_nowait()
    assert (gen, voice, text, pre_pcm) == (0, "leijun", "", err_pcm)
    bot._tts_queue.task_done()

    # 2. Failed TTS synthesis pushes cached error cue PCM onto _play_queue
    class BrokenTTS:
        async def synthesize(self, s: str) -> bytes:
            raise RuntimeError("tts boom")

    bot.tts = BrokenTTS()
    await bot._tts_queue.put((0, "leijun", "测试失败", None))
    worker = asyncio.create_task(bot._synth_worker())
    queued_gen, queued_pcm = await asyncio.wait_for(bot._play_queue.get(), timeout=1.0)
    assert (queued_gen, queued_pcm) == (0, err_pcm)
    bot._play_queue.task_done()
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass


async def test_on_speaker_connected_is_silent():
    from kesoku.gateway.chatbot.xiaoai.adapter import SpeakerState

    bot = _make_stub_xiaoai_bot()
    bot._all_wake_words = ["小雷小雷"]
    bot.state = SpeakerState.LISTENING
    await bot._on_speaker_connected()

    assert bot.state == SpeakerState.IDLE
    assert bot._test_state.get_kws_resets() == 1
    assert bot._test_state.get_vad_resets() == 1
    assert bot._test_state.played_pcms == []


async def test_ws_server_local_kws_recording_control_and_http():
    import json
    from types import SimpleNamespace

    sent_rpcs: list[str] = []

    class FakeWS:
        remote_address = ("192.168.1.40", 12345)

        async def send(self, data: str | bytes) -> None:
            if isinstance(data, str):
                msg = json.loads(data)
                if "Request" in msg:
                    sent_rpcs.append(msg["Request"]["command"])

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    triggered: list[str] = []

    async def on_wakeup(keyword: str) -> str:
        triggered.append(keyword)
        return "ok"

    server = OpenXiaoAiServer(auto_start_recording=False, on_local_wakeup=on_wakeup)

    # Disconnected -> 503 Service Unavailable; missing keyword -> 400; other paths -> None
    assert await server._process_http_request(None, SimpleNamespace(path="/")) is None
    resp_400 = await server._process_http_request(None, SimpleNamespace(path="/local-wakeup"))
    assert resp_400.status_code == 400
    resp_503 = await server._process_http_request(
        None, SimpleNamespace(path="/local-wakeup?keyword=%E5%B0%8F%E5%BC%A0%E5%B0%8F%E5%BC%A0")
    )
    assert resp_503.status_code == 503

    # Connecting with auto_start_recording=False skips start_recording
    await server._handle_connection(FakeWS())
    assert sent_rpcs == ["start_play"]
    assert server.connected is False

    # Connected + start/stop recording idempotency + 200 OK /local-wakeup
    sent_rpcs.clear()
    server._ws = FakeWS()
    resp_200 = await server._process_http_request(
        None, SimpleNamespace(path="/local-wakeup?keyword=%E5%B0%8F%E5%BC%A0%E5%B0%8F%E5%BC%A0")
    )
    assert resp_200.status_code == 200
    assert json.loads(resp_200.body) == {"status": "ok"}
    assert triggered == ["小张小张"]

    await server.start_recording()
    await server.start_recording()  # idempotent
    await server.stop_recording()
    await server.stop_recording()  # idempotent
    assert sent_rpcs == ["start_recording", "stop_recording"]


async def test_chatbot_local_kws_lifecycle():
    import asyncio
    from types import SimpleNamespace

    from kesoku.gateway.chatbot.xiaoai.adapter import SpeakerState

    assert XiaoAiConfig().local_kws is False

    bot = _make_stub_xiaoai_bot()
    bot.config = XiaoAiConfig(enabled=True, local_kws=True, continuous_listen_sec=8.0)
    bot._local_wakeup_task = None
    bot._wake_word_to_role = {"小雷小雷": "leijun", "小张小张": "xiaozhang"}
    bot._personas = {"leijun": object(), "xiaozhang": object()}
    bot._persona_cues["xiaozhang"] = VoiceCueBank()
    bot.tts = SimpleNamespace(set_active_voice=lambda r: None)

    rec_calls: list[str] = []

    async def fake_start_rec() -> None:
        rec_calls.append("start")

    async def fake_stop_rec() -> None:
        rec_calls.append("stop")

    bot.ws_server.start_recording = fake_start_rec
    bot.ws_server.stop_recording = fake_stop_rec

    # Local wakeup in IDLE schedules activation -> starts recording & enters LISTENING
    assert await bot._on_local_wakeup("小张小张") == "ok"
    assert bot._local_wakeup_task is not None
    await bot._local_wakeup_task
    assert rec_calls == ["start"]
    assert bot.state == SpeakerState.LISTENING
    assert bot._active_role == "xiaozhang"

    # Subsequent local wakeup while active is ignored
    assert await bot._on_local_wakeup("小雷小雷") == "active"

    # Stop command returns to IDLE and stops recording
    await bot._handle_stop_command("闭嘴")
    assert bot.state == SpeakerState.IDLE
    assert rec_calls == ["start", "stop"]

    # Watchdog expiry from LISTENING returns to IDLE and stops recording
    assert await bot._on_local_wakeup("小雷小雷") == "ok"
    await bot._local_wakeup_task
    assert bot.state == SpeakerState.LISTENING
    bot._listen_deadline = 0.0
    watchdog = asyncio.create_task(bot._listen_watchdog())
    await asyncio.sleep(0.65)
    watchdog.cancel()
    try:
        await watchdog
    except asyncio.CancelledError:
        pass
    assert bot.state == SpeakerState.IDLE
    assert rec_calls == ["start", "stop", "start", "stop"]


async def test_turn_id_isolation_drops_stale_stream_and_final_messages():
    from kesoku.constants import MessageRole, MessageStatus, MessageType
    from kesoku.db import Message

    bot = _make_stub_xiaoai_bot(progress_feedback="status")
    bot._personas = {"leijun": object(), "xiaozhang": object()}
    bot._persona_cues["xiaozhang"] = VoiceCueBank()
    bot._active_role = "xiaozhang"
    bot._active_turn_channel_id = "xiaoai:xiaozhang"
    bot._active_turn_id = "turn_2"
    bot._aborted_turn_ids = {"turn_1"}
    bot._turn_in_progress = True

    # Stale turn_1 stream delta, tool cue, and final message are all dropped without clobbering turn_2
    await bot._handle_stream_update("xiaoai:xiaozhang", "s2", "turn_1", "这是一句迟到的旧回复。")
    assert bot._tts_queue.empty()

    stale_tc = Message(
        id="tc_stale",
        session_id="s1",
        chatbot_id="xiaoai",
        channel_id="xiaoai:xiaozhang",
        sender="Kesoku",
        role=MessageRole.TOOL,
        type=MessageType.TOOL_CALL,
        content="Calling tool `web_search`",
        parent_id="turn_1",
        metadata={"tool_name": "web_search"},
    )
    await bot.handle_message(stale_tc)
    assert bot._tts_queue.empty()

    stale_final = Message(
        id="msg_stale_final",
        session_id="s1",
        chatbot_id="xiaoai",
        channel_id="xiaoai:xiaozhang",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="旧问题最终答复。",
        parent_id="turn_1",
    )
    await bot.handle_message(stale_final)
    assert bot._tts_queue.empty()
    assert bot._turn_in_progress is True

    # Cross-persona message from xiaoai:leijun while xiaoai:xiaozhang is active is also dropped
    cross_persona_final = Message(
        id="msg_cross_final",
        session_id="s_lei",
        chatbot_id="xiaoai",
        channel_id="xiaoai:leijun",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="雷军晚到的回复。",
        parent_id="turn_lei_9",
    )
    await bot.handle_message(cross_persona_final)
    assert bot._tts_queue.empty()
    assert bot._turn_in_progress is True
    assert bot._test_state.delivered == [
        ("tc_stale", MessageStatus.DELIVERED),
        ("msg_stale_final", MessageStatus.DELIVERED),
        ("msg_cross_final", MessageStatus.DELIVERED),
    ]

    # Active turn_2 stream + final reply render normally and finish the turn cleanly
    await bot._handle_stream_update("xiaoai:xiaozhang", "s2", "turn_2", "你好，这是当前回复。")
    assert bot._tts_queue.qsize() == 1
    gen, voice, text, pre_pcm = bot._tts_queue.get_nowait()
    assert (gen, voice, text, pre_pcm) == (0, "xiaozhang", "你好，这是当前回复。", None)
    bot._tts_queue.task_done()

    active_final = Message(
        id="msg_cur_final",
        session_id="s2",
        chatbot_id="xiaoai",
        channel_id="xiaoai:xiaozhang",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="你好，这是当前回复。",
        parent_id="turn_2",
    )
    await bot.handle_message(active_final)
    assert bot._turn_in_progress is False
    assert bot._active_turn_id is None


async def test_barge_in_persona_switch_and_clear_abort_all_channels():
    from types import SimpleNamespace

    from kesoku.constants import MessageStatus
    from kesoku.gateway.chatbot.xiaoai.adapter import SpeakerState

    bot = _make_stub_xiaoai_bot()
    bot._personas = {"leijun": object(), "xiaozhang": object()}
    bot._wake_word_to_role = {"小雷小雷": "leijun", "小张小张": "xiaozhang"}
    bot._all_wake_words = ["小雷小雷", "小张小张"]
    bot._persona_cues["xiaozhang"] = VoiceCueBank()
    bot.tts = SimpleNamespace(set_active_voice=lambda r: None)

    abort_calls: list[tuple[str, bool]] = []
    cleared_channels: list[str] = []
    ingested: list[object] = []

    async def multi_get_session(chatbot_id: str, channel_id: str):
        mapping = {
            "xiaoai:leijun": SimpleNamespace(id="sess_lei"),
            "xiaoai:xiaozhang": SimpleNamespace(id="sess_zhang"),
            "xiaoai:default": SimpleNamespace(id="sess_default"),
        }
        return mapping.get(channel_id)

    async def multi_abort(session_id: str, *, skip_memory_close: bool = False) -> None:
        abort_calls.append((session_id, skip_memory_close))

    async def fake_clear(channel_id: str) -> bool:
        cleared_channels.append(channel_id)
        return True

    async def fake_compile() -> str:
        return "prompt"

    async def fake_reset_idle(channel_id: str | None = None) -> bool:
        return False

    async def fake_ingest(dto) -> bool:
        ingested.append(dto)
        await bot.post_ingest_hook(
            SimpleNamespace(id="sess_zhang"),
            SimpleNamespace(id="turn_new_2", channel_id=dto.channel_id),
            dto,
        )
        return False

    bot.gateway.db.get_session_by_channel = multi_get_session
    bot.gateway.abort_session = multi_abort
    bot.clear_session_by_channel = fake_clear
    bot._compile_custom_prompt = fake_compile
    bot._maybe_reset_idle_session = fake_reset_idle
    bot.ingest_message = fake_ingest

    # 1. Wake-word barge-in from THINKING aborts ALL persona channels & records old turn_id
    bot.state = SpeakerState.THINKING
    bot._active_turn_id = "turn_old_1"
    bot._active_turn_channel_id = "xiaoai:leijun"
    bot._turn_in_progress = True
    await bot._handle_wakeup("小张小张")
    assert ("sess_lei", True) in abort_calls
    assert ("sess_zhang", True) in abort_calls
    assert ("sess_default", True) in abort_calls
    assert "turn_old_1" in bot._aborted_turn_ids
    assert bot._active_role == "xiaozhang"

    # 2. One-breath persona switch ("小雷小雷今天天气怎么样") interrupts & starts new turn on leijun
    abort_calls.clear()
    bot.pipeline.transcribe = lambda arr: "小雷小雷 今天天气怎么样"
    await bot._handle_utterance(np.zeros(1600, dtype=np.float32))
    assert bot._active_role == "leijun"
    assert ("sess_lei", True) in abort_calls
    assert ("sess_zhang", True) in abort_calls
    assert len(ingested) == 1
    assert ingested[0].channel_id == "xiaoai:leijun"
    assert bot._active_turn_channel_id == "xiaoai:leijun"
    assert bot._active_turn_id == "turn_new_2"
    assert bot._turn_in_progress is True

    # 3. Clear / new-conversation command interrupts active work and clears current channel session
    abort_calls.clear()
    assert await bot._try_local_command("开新对话") is True
    assert ("sess_lei", True) in abort_calls
    assert "turn_new_2" in bot._aborted_turn_ids
    assert cleared_channels == ["xiaoai:leijun"]
    assert bot._tts_queue.qsize() == 1
    assert bot._tts_queue.get_nowait()[2] == "好的，已经开启新对话。"
    bot._tts_queue.task_done()
    assert MessageStatus.DELIVERED is not None


async def test_thinking_timeout_aborts_and_blocks_late_reply():
    import asyncio
    import time

    from kesoku.constants import MessageRole, MessageType
    from kesoku.db import Message
    from kesoku.gateway.chatbot.xiaoai.adapter import SpeakerState

    bot = _make_stub_xiaoai_bot()
    err_pcm = b"\x02\x00" * 480
    bot._persona_cues["leijun"]._clips["error"] = [err_pcm]

    bot.state = SpeakerState.THINKING
    bot._turn_in_progress = True
    bot._turn_start_ts = time.monotonic() - 65.0
    bot._active_turn_id = "turn_stalled_99"
    bot._active_turn_channel_id = "xiaoai:default"

    watchdog = asyncio.create_task(bot._listen_watchdog())
    await asyncio.sleep(0.65)
    watchdog.cancel()
    try:
        await watchdog
    except asyncio.CancelledError:
        pass

    assert bot._test_state.aborted == ["sess_active"]
    assert "turn_stalled_99" in bot._aborted_turn_ids
    assert bot._turn_in_progress is False
    assert bot._tts_queue.qsize() == 1
    assert bot._tts_queue.get_nowait()[3] == err_pcm
    bot._tts_queue.task_done()

    # Late response for timed-out turn is dropped cleanly
    late_msg = Message(
        id="msg_late_99",
        session_id="sess_active",
        chatbot_id="xiaoai",
        channel_id="xiaoai:default",
        sender="Kesoku",
        role=MessageRole.ASSISTANT,
        type=MessageType.TEXT,
        content="超时后才返回的旧回复。",
        parent_id="turn_stalled_99",
    )
    await bot.handle_message(late_msg)
    assert bot._tts_queue.empty()


async def test_gateway_abort_session_forwards_skip_memory_close(tmp_path):
    from unittest.mock import AsyncMock

    from kesoku.config import KesokuConfig, WorkspaceConfig
    from kesoku.context import KesokuContext
    from kesoku.db import DatabaseManager
    from kesoku.gateway.gateway import Gateway

    db_file = str(tmp_path / "gw_abort.db")
    DatabaseManager(db_file).init_tables()
    gw = Gateway(context=KesokuContext(config=KesokuConfig(workspace=WorkspaceConfig(db_path=db_file))))
    mock_agent = AsyncMock()
    mock_agent.stop_session_worker = AsyncMock()
    gw.register_agent(mock_agent)

    await gw.abort_session("sess_xyz", skip_memory_close=True)
    mock_agent.stop_session_worker.assert_awaited_once_with(
        "sess_xyz", immediate=True, skip_memory_close=True
    )


async def test_continuous_listen_sec_3s_not_clamped_to_8s():
    from types import SimpleNamespace

    bot = _make_stub_xiaoai_bot()
    bot.config = XiaoAiConfig(enabled=True, continuous_listen_sec=3.0)
    bot._wake_word_to_role = {"小张小张": "xiaozhang"}
    bot._personas = {"xiaozhang": object()}
    bot._persona_cues["xiaozhang"] = VoiceCueBank()
    bot.tts = SimpleNamespace(set_active_voice=lambda r: None)

    await bot._handle_wakeup("小张小张")
    wake_window = bot._listen_deadline - bot._vad_unmute_ts
    assert 2.9 <= wake_window <= 3.1

    bot.config = XiaoAiConfig(enabled=True, continuous_listen_sec=0.0)
    await bot._handle_wakeup("小张小张")
    assert 5.9 <= (bot._listen_deadline - bot._vad_unmute_ts) <= 6.1


def test_xiaoai_config_compact_prompt_and_allowed_tools():
    default_cfg = XiaoAiConfig()
    assert default_cfg.compact_system_prompt is False
    assert default_cfg.allowed_tools is None

    custom_cfg = XiaoAiConfig(
        enabled=True,
        compact_system_prompt=True,
        allowed_tools=["web_search"],
    )
    assert custom_cfg.compact_system_prompt is True
    assert custom_cfg.allowed_tools == ["web_search"]


def test_build_sys_prompt_compact_omits_desktop_boilerplate(tmp_path, monkeypatch):
    from kesoku.agent.prompt import build_sys_prompt, is_compact_prompt_chatbot
    from kesoku.db import Session

    role_dir = tmp_path / "roles" / "xiaozhang"
    role_dir.mkdir(parents=True)
    (role_dir / "intro.md").write_text("# 核心人设\n你是家庭语音助手小张。", encoding="utf-8")

    cfg = KesokuConfig(
        chatbots={
            "xiaoai": [
                XiaoAiConfig(
                    enabled=True,
                    chatbot_id="xiaoai",
                    default_role="xiaozhang",
                    compact_system_prompt=True,
                    allowed_tools=["web_search"],
                )
            ]
        },
        agent_working_dir=str(tmp_path),
    )
    monkeypatch.setattr("kesoku.agent.prompt.get_config", lambda: cfg)

    assert is_compact_prompt_chatbot(cfg, "xiaoai") is True
    assert is_compact_prompt_chatbot(cfg, "discord") is False

    sess = Session(id="s_compact", title="Compact", workspace_name="ws_compact", role_name="xiaozhang")
    speaker_guide = "# 智能音箱语音交互指南\n- 简洁口语回复，1~3句以内。"
    compact_prompt = build_sys_prompt(
        custom_prompt=speaker_guide,
        session=sess,
        role="xiaozhang",
        compact=True,
    )

    assert "你是家庭语音助手小张" in compact_prompt
    assert speaker_guide in compact_prompt
    assert f"AWD='{tmp_path}'" in compact_prompt
    assert "# Memory Rules" in compact_prompt
    for omitted in (
        "# Output Formatting Rules",
        "# Pinned Session Notes and Context Blocks",
        "# File Tools",
        "# Background Tasks",
        "# Agent Working Directory",
        "# Session Staging Directory",
        "# Available Skills",
    ):
        assert omitted not in compact_prompt
    assert len(compact_prompt) < 2000


def test_allowed_tools_filtering_in_registry_and_turn_executor():
    from kesoku.agent.tools.registry import ToolRegistry
    from kesoku.agent.turn_executor import TurnExecutor
    from kesoku.context import KesokuContext
    from kesoku.db import Message

    reg = ToolRegistry()

    @reg.register
    def web_search(query: str) -> str:
        """Search the web."""
        return query

    @reg.register
    def read_file(file_name: str) -> str:
        """Read a file."""
        return file_name

    @reg.register
    def run_shell_command(cmd: str) -> str:
        """Run shell."""
        return cmd

    @reg.register
    def update_file(file_name: str, new_content: str) -> str:
        """Update a file."""
        return file_name

    assert [f.__name__ for f in reg.get_tools_list()] == [
        "web_search",
        "read_file",
        "run_shell_command",
        "update_file",
    ]
    assert [f.__name__ for f in reg.get_tools_list(allowed=["web_search"])] == ["web_search"]
    assert [
        f.__name__
        for f in reg.get_tools_list(allowed=["web_search", "read_file", "run_shell_command"])
    ] == ["web_search", "read_file", "run_shell_command"]

    cfg = KesokuConfig(
        chatbots={
            "xiaoai": [
                XiaoAiConfig(
                    enabled=True,
                    chatbot_id="xiaoai",
                    allowed_tools=["web_search", "read_file", "run_shell_command"],
                )
            ]
        }
    )
    te = TurnExecutor(session_id="s1", gateway=None, tool_runner=None, context=KesokuContext(config=cfg))
    xiaoai_msg = Message(
        id="m1", session_id="s1", chatbot_id="xiaoai", channel_id="xiaoai:xiaozhang", sender="u", content="hi"
    )
    discord_msg = Message(id="m2", session_id="s2", chatbot_id="discord", channel_id="c1", sender="u", content="hi")
    assert te._resolve_allowed_tools(xiaoai_msg) == ["web_search", "read_file", "run_shell_command"]
    assert te._resolve_allowed_tools(discord_msg) is None


async def test_ws_server_mic_mute_http_and_connection():
    import json
    from types import SimpleNamespace

    sent_rpcs: list[str] = []

    class FakeWS:
        remote_address = ("192.168.1.40", 12345)

        async def send(self, data: str | bytes) -> None:
            if isinstance(data, str):
                msg = json.loads(data)
                if "Request" in msg:
                    sent_rpcs.append(msg["Request"]["command"])

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    mute_events: list[bool] = []

    async def on_mute(muted: bool) -> str:
        mute_events.append(muted)
        return "muted" if muted else "unmuted"

    server = OpenXiaoAiServer(auto_start_recording=True, on_mic_mute=on_mute)

    # Missing or invalid muted query parameter -> 400
    assert (await server._process_http_request(None, SimpleNamespace(path="/mic-mute"))).status_code == 400
    assert (
        await server._process_http_request(None, SimpleNamespace(path="/mic-mute?muted=maybe"))
    ).status_code == 400

    # /mic-mute?muted=1 works even before speaker WS connects and prevents auto-start recording
    resp_mute = await server._process_http_request(None, SimpleNamespace(path="/mic-mute?muted=1"))
    assert resp_mute.status_code == 200
    assert json.loads(resp_mute.body) == {"status": "muted", "muted": True}
    assert server.mic_muted is True
    assert mute_events == [True]

    await server._handle_connection(FakeWS())
    assert sent_rpcs == ["start_play"]

    # Connected while unmuted -> /mic-mute?muted=0 triggers unmute callback and future connect auto-starts
    sent_rpcs.clear()
    resp_unmute = await server._process_http_request(None, SimpleNamespace(path="/mic-mute?muted=0"))
    assert resp_unmute.status_code == 200
    assert json.loads(resp_unmute.body) == {"status": "unmuted", "muted": False}
    assert server.mic_muted is False
    assert mute_events == [True, False]

    await server._handle_connection(FakeWS())
    assert sent_rpcs == ["start_recording", "start_play"]

    # force=True sends stop_recording even if _recording_active was tracked False
    sent_rpcs.clear()
    server._ws = FakeWS()
    server._recording_active = False
    await server.stop_recording(force=True)
    assert sent_rpcs == ["stop_recording"]


async def test_chatbot_mic_mute_stops_and_resumes_recording():
    from kesoku.gateway.chatbot.xiaoai.adapter import SpeakerState

    bot = _make_stub_xiaoai_bot()
    bot.config = XiaoAiConfig(enabled=True, local_kws=False)

    rec_calls: list[tuple[str, bool]] = []

    async def fake_start_rec() -> None:
        rec_calls.append(("start", False))

    async def fake_stop_rec(*, force: bool = False) -> None:
        rec_calls.append(("stop", force))

    bot.ws_server.start_recording = fake_start_rec
    bot.ws_server.stop_recording = fake_stop_rec
    bot.ws_server.mic_muted = False

    # Muting during SPEAKING + active turn aborts turn, resets state to IDLE, and stops recording forcefully
    bot.state = SpeakerState.SPEAKING
    bot._turn_in_progress = True
    bot._active_turn_id = "turn_muted_1"
    await bot._tts_queue.put((0, "leijun", "没说完", None))
    assert await bot._on_mic_mute(True) == "muted"
    assert bot._mic_mute_task is not None
    await bot._mic_mute_task

    assert bot._mic_muted is True
    assert bot.ws_server.mic_muted is True
    assert bot.state == SpeakerState.IDLE
    assert bot._turn_in_progress is False
    assert "turn_muted_1" in bot._aborted_turn_ids
    assert bot._test_state.aborted == ["sess_active"]
    assert bot._test_state.get_stopped_play() == 1
    assert rec_calls == [("stop", True)]

    # Residual audio frames & local wakeup triggers are ignored while muted
    kws_called = False

    def spy_kws(samples):
        nonlocal kws_called
        kws_called = True
        return "小张小张"

    bot.pipeline.feed_kws = spy_kws
    await bot._on_audio_frame(np.ones(512, dtype=np.float32))
    assert kws_called is False
    assert await bot._on_local_wakeup("小张小张") == "muted"

    # Unmuting when local_kws=False resumes start_recording
    assert await bot._on_mic_mute(False) == "unmuted"
    await bot._mic_mute_task
    assert bot._mic_muted is False
    assert bot.ws_server.mic_muted is False
    assert rec_calls == [("stop", True), ("start", False)]

    # When local_kws=True, unmuting stays IDLE without auto-starting recording
    rec_calls.clear()
    bot.config = XiaoAiConfig(enabled=True, local_kws=True)
    assert await bot._on_mic_mute(True) == "muted"
    await bot._mic_mute_task
    assert await bot._on_mic_mute(False) == "unmuted"
    await bot._mic_mute_task
    assert rec_calls == [("stop", True)]


async def test_led_feedback_syncs_native_effects_with_state():
    from types import SimpleNamespace

    from kesoku.gateway.chatbot.xiaoai.adapter import SpeakerState

    assert XiaoAiConfig().led_feedback is True

    shell_cmds: list[str] = []

    async def fake_run_shell(cmd: str) -> None:
        shell_cmds.append(cmd)

    bot = _make_stub_xiaoai_bot()
    bot._wake_word_to_role = {"小张小张": "xiaozhang"}
    bot._all_wake_words = ["小张小张"]
    bot._personas = {"xiaozhang": object()}
    bot._persona_cues["xiaozhang"] = VoiceCueBank()
    bot.tts = SimpleNamespace(set_active_voice=lambda r: None)
    bot.ws_server.run_shell = fake_run_shell

    # Connecting clears managed LEDs 1/2/3 without touching mute LED 7
    await bot._on_speaker_connected()
    assert shell_cmds == [
        "/bin/shut_led 1 >/dev/null 2>&1; /bin/shut_led 2 >/dev/null 2>&1; /bin/shut_led 3 >/dev/null 2>&1"
    ]
    assert "7" not in shell_cmds[0]

    # Wakeup -> LISTENING (effect 1)
    shell_cmds.clear()
    await bot._handle_wakeup("小张小张")
    assert shell_cmds == ["/bin/show_led 1 >/dev/null 2>&1"]
    assert bot._active_led == 1

    # THINKING -> effect 2 on, effect 1 off
    shell_cmds.clear()
    await bot._set_state(SpeakerState.THINKING)
    assert shell_cmds == ["/bin/show_led 2 >/dev/null 2>&1; /bin/shut_led 1 >/dev/null 2>&1"]
    assert bot._active_led == 2

    # SPEAKING -> effect 3 on, effect 2 off; repeated SPEAKING deduplicated
    shell_cmds.clear()
    await bot._set_state(SpeakerState.SPEAKING)
    await bot._set_state(SpeakerState.SPEAKING)
    assert shell_cmds == ["/bin/show_led 3 >/dev/null 2>&1; /bin/shut_led 2 >/dev/null 2>&1"]
    assert bot._active_led == 3

    # Returning to IDLE -> shuts managed effects 1/2/3 cleanly
    shell_cmds.clear()
    await bot._enter_idle()
    assert shell_cmds == [
        "/bin/shut_led 1 >/dev/null 2>&1; /bin/shut_led 2 >/dev/null 2>&1; /bin/shut_led 3 >/dev/null 2>&1"
    ]
    assert bot._active_led is None

    # Muted mic suppresses show_led while keeping force clear
    shell_cmds.clear()
    bot._mic_muted = True
    await bot._set_state(SpeakerState.LISTENING)
    assert shell_cmds == []
    assert bot._active_led is None

    # Disabled led_feedback emits no shell commands
    bot._mic_muted = False
    bot.config = XiaoAiConfig(enabled=True, led_feedback=False)
    await bot._set_state(SpeakerState.LISTENING, force_led=True)
    assert shell_cmds == []
