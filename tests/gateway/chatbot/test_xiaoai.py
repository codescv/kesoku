"""Unit tests for the Open-XiaoAI smart speaker adapter components."""

import numpy as np

from kesoku.config import KesokuConfig, XiaoAiConfig
from kesoku.gateway.chatbot.xiaoai.audio_pipeline import chinese_phrase_to_kws_tokens
from kesoku.gateway.chatbot.xiaoai.tts import (
    MlxQwenTTSEngine,
    SentenceChunker,
    VoiceCueBank,
    clean_text_for_speech,
    float32_to_pcm16,
    make_wakeup_chime,
    trim_pcm16_silence,
)
from kesoku.gateway.chatbot.xiaoai.ws_server import decode_a113_s32_to_s16


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
    for name, amp in [("a.wav", 1500), ("b.wav", 2500)]:
        p = wakeup_dir / name
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(np.full(1200, amp, dtype="<i2").tobytes())

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
