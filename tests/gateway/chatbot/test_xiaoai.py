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
    bot._turn_in_progress = False
    bot._suppress_until_next_user_turn = False
    bot._active_stream_turn_id = None
    bot._streamed_any_sentence = False
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
