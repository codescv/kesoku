"""Unit tests for history preparation (truncation, determinism, historical question-option stripping)."""

from kesoku.agent.history import prepare_history_for_llm, strip_question_options
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message


def test_truncate_tool_call_arguments_only_for_file_tools():
    """Long string args of update_session_notes/update_file are truncated; other tools are untouched."""
    from kesoku.agent.history import MAX_HISTORICAL_TOOL_ARG_CHARS, truncate_tool_call_arguments

    long = "n" * 2000
    args = {"content": long}
    out = truncate_tool_call_arguments("update_session_notes", args)
    assert out is not args
    assert out["content"].startswith("n" * MAX_HISTORICAL_TOOL_ARG_CHARS)
    assert "2000 chars total" in out["content"]
    assert args["content"] == long  # original not mutated
    img_args = {"command": "uv run generate_image.py --prompt " + long}
    assert truncate_tool_call_arguments("run_shell_command", img_args) is img_args
    short = {"content": "short"}
    assert truncate_tool_call_arguments("update_file", short) is short


def test_prepare_history_truncates_completed_notes_tool_call_args():
    """Completed turns get truncated notes arguments; the active turn keeps them in full."""
    from kesoku.agent.history import prepare_history_for_llm
    from kesoku.constants import MessageRole, MessageStatus, MessageType
    from kesoku.db import Message

    def call(i: int) -> list[Message]:
        return [
            Message(session_id="s", chatbot_id="cb", channel_id="ch", sender="u", role=MessageRole.USER,
                    type=MessageType.TEXT, content=f"turn {i}", status=MessageStatus.RESPONDED, timestamp=i * 10),
            Message(session_id="s", chatbot_id="cb", channel_id="ch", sender="update_session_notes",
                    role=MessageRole.TOOL, type=MessageType.TOOL_CALL, content="Calling tool",
                    status=MessageStatus.RESPONDED, timestamp=i * 10 + 1,
                    metadata={"tool_name": "update_session_notes", "tool_arguments": {"content": "n" * 2000}}),
        ]

    prepared = prepare_history_for_llm(call(1) + call(2))
    calls = [m for m in prepared if m.type == MessageType.TOOL_CALL]
    assert len(calls[0].metadata["tool_arguments"]["content"]) < 400
    assert len(calls[1].metadata["tool_arguments"]["content"]) == 2000


def test_prepare_history_truncation_is_deterministic(tmp_path):
    """Truncated historical tool output must be byte-identical across calls (required for implicit caching)."""
    import time as _time

    from kesoku.agent.history import prepare_history_for_llm
    from kesoku.constants import MessageRole, MessageStatus, MessageType
    from kesoku.db import Message

    def turn(i: int) -> list[Message]:
        return [
            Message(id=f"u{i}", session_id="s", chatbot_id="cb", channel_id="ch", sender="u", role=MessageRole.USER,
                    type=MessageType.TEXT, content=f"q{i}", status=MessageStatus.RESPONDED, timestamp=i * 10),
            Message(id=f"r{i}", session_id="s", chatbot_id="cb", channel_id="ch", sender="chat_search",
                    role=MessageRole.TOOL, type=MessageType.TOOL_RESULT, content="x" * 2000,
                    status=MessageStatus.RESPONDED, timestamp=i * 10 + 1,
                    metadata={"tool_name": "chat_search", "tool_result": "x" * 2000}),
        ]

    history = turn(1) + turn(2)
    first = prepare_history_for_llm(history, staging_dir=str(tmp_path))
    _time.sleep(1.1)
    second = prepare_history_for_llm(history, staging_dir=str(tmp_path))
    assert [m.content for m in first] == [m.content for m in second]
    assert "hist_tool_chat_search_r1.txt" in first[1].content


# ---------------------------------------------------------------------------
# Historical [question: ...] option stripping (generic engine syntax from prompt.py)
# ---------------------------------------------------------------------------

Q_BLOCK = "[question: 下一步怎么做？ || 像香槟软木塞一样弹射出去 | 冷静地递上一杯热可可 | 拉着队友逃跑]"
STRIPPED = "[question: ...]"


def _hmsg(role: MessageRole, content: str, ts: float, msg_id: str, msg_type: MessageType = MessageType.TEXT) -> Message:
    """Build a minimal history Message."""
    return Message(
        id=msg_id,
        session_id="s",
        chatbot_id="cli",
        channel_id="ch1",
        sender="u1" if role == MessageRole.USER else "Kesoku",
        role=role,
        type=msg_type,
        content=content,
        status=MessageStatus.RESPONDED,
        timestamp=ts,
    )


def _question_history(num_completed: int, with_active: bool = True) -> list[Message]:
    """Build `num_completed` user/assistant turns whose assistant reply ends with a [question:] block."""
    history: list[Message] = []
    for i in range(num_completed):
        history.append(_hmsg(MessageRole.USER, f"用户第{i}轮", i * 100.0, f"u{i}"))
        block = Q_BLOCK.replace("下一步", f"第{i}步")
        history.append(_hmsg(MessageRole.ASSISTANT, f"第{i}轮的正文。\n{block}", i * 100.0 + 10, f"a{i}"))
    if with_active:
        history.append(_hmsg(MessageRole.USER, "冷静地递上一杯热可可", num_completed * 100.0, "u_active"))
    return history


def _assistant_texts(prepared: list[Message]) -> list[str]:
    """Contents of assistant messages, in order."""
    return [m.content for m in prepared if m.role == MessageRole.ASSISTANT]


def test_strip_question_options_basic() -> None:
    """Stem and options collapse into the placeholder; surrounding text is preserved."""
    out = strip_question_options(f"正文。\n{Q_BLOCK}")
    assert out == f"正文。\n{STRIPPED}"
    assert "||" not in out and "香槟" not in out


def test_strip_question_options_multiple_idempotent_and_plain() -> None:
    """All blocks are stripped, stripping is idempotent, and text without blocks is untouched."""
    out = strip_question_options(f"A {Q_BLOCK} B [question: 选哪个？ || x | y] C")
    assert out == f"A {STRIPPED} B {STRIPPED} C"
    assert strip_question_options(out) == out
    plain = "普通回复 [file: a.png] [voice: b.mp3]"
    assert strip_question_options(plain) == plain
    assert strip_question_options("") == ""


def test_strip_question_options_keeps_attachment_markup() -> None:
    """[file:]/[voice:] markup adjacent to a question block survives."""
    out = strip_question_options(f"[file: images/x.png]\n台词\n{Q_BLOCK}\n[voice: audio/a.mp3]")
    assert "[file: images/x.png]" in out and "[voice: audio/a.mp3]" in out and STRIPPED in out


def test_prepare_history_strips_turns_older_than_last_completed() -> None:
    """Turns with idx < num_turns - 2 are stripped; the last completed turn and the active turn are intact."""
    prepared = prepare_history_for_llm(_question_history(4), strip_old_questions=True)
    texts = _assistant_texts(prepared)
    assert len(texts) == 4
    for t in texts[:3]:
        assert STRIPPED in t and "||" not in t and "香槟" not in t
        assert "轮的正文" in t
    assert "第3步" in texts[3] and "||" in texts[3] and "像香槟软木塞一样弹射出去" in texts[3]


def test_prepare_history_without_active_turn() -> None:
    """Without an active user turn the final turn is 'latest' and the one before it the last completed."""
    texts = _assistant_texts(prepare_history_for_llm(_question_history(3, with_active=False), strip_old_questions=True))
    assert STRIPPED in texts[0] and "||" not in texts[0]
    assert "||" in texts[1] and "||" in texts[2]


def test_prepare_history_short_history_untouched() -> None:
    """One completed turn plus the active turn: nothing is stripped."""
    texts = _assistant_texts(prepare_history_for_llm(_question_history(1), strip_old_questions=True))
    assert "||" in texts[0]


def test_prepare_history_question_flag_off() -> None:
    """strip_old_questions=False keeps every option."""
    texts = _assistant_texts(prepare_history_for_llm(_question_history(4), strip_old_questions=False))
    assert all("||" in t for t in texts)


def test_prepare_history_question_strip_does_not_mutate_input() -> None:
    """Originals are untouched and user messages keep their text."""
    history = _question_history(4)
    before = [m.content for m in history]
    prepared = prepare_history_for_llm(history, strip_old_questions=True)
    assert [m.content for m in history] == before
    users = [m.content for m in prepared if m.role == MessageRole.USER]
    assert any("用户第0轮" in u for u in users)
    assert any("冷静地递上一杯热可可" in u for u in users)


def test_prepare_history_question_strip_prefix_stable() -> None:
    """Already-stripped turns render byte-identically as new turns arrive (implicit prefix caching)."""
    p1 = _assistant_texts(prepare_history_for_llm(_question_history(4), strip_old_questions=True))
    p2 = _assistant_texts(prepare_history_for_llm(_question_history(5), strip_old_questions=True))
    assert p1[:3] == p2[:3]


def test_prepare_history_question_strip_skips_tool_messages() -> None:
    """Tool results are not rewritten by the question stripper (existing truncation logic owns them)."""
    history = _question_history(3)
    history.insert(
        1,
        Message(
            id="t0", session_id="s", chatbot_id="cli", channel_id="ch1", sender="chat_search",
            role=MessageRole.TOOL, type=MessageType.TOOL_RESULT, content=f"Tool returned: {Q_BLOCK}",
            status=MessageStatus.RESPONDED, timestamp=5.0,
            metadata={"tool_name": "chat_search", "tool_result": Q_BLOCK},
        ),
    )
    prepared = prepare_history_for_llm(history, strip_old_questions=True)
    tool_msgs = [m for m in prepared if m.role == MessageRole.TOOL]
    assert tool_msgs and "||" in tool_msgs[0].content


def test_prepare_history_truncates_completed_update_file_args() -> None:
    """Completed-turn update_file calls (notes edits) get old/new content truncated; the active turn is kept."""

    def call(i: int) -> list[Message]:
        return [
            Message(session_id="s", chatbot_id="cb", channel_id="ch", sender="u", role=MessageRole.USER,
                    type=MessageType.TEXT, content=f"turn {i}", status=MessageStatus.RESPONDED, timestamp=i * 10),
            Message(session_id="s", chatbot_id="cb", channel_id="ch", sender="update_file",
                    role=MessageRole.TOOL, type=MessageType.TOOL_CALL, content="Calling tool",
                    status=MessageStatus.RESPONDED, timestamp=i * 10 + 1,
                    metadata={"tool_name": "update_file", "tool_arguments": {
                        "file_name": "/tmp/s/notes.md", "old_content": "o" * 2000, "new_content": "n" * 2000}}),
        ]

    prepared = prepare_history_for_llm(call(1) + call(2))
    calls = [m for m in prepared if m.type == MessageType.TOOL_CALL]
    old_args, new_args = calls[0].metadata["tool_arguments"], calls[1].metadata["tool_arguments"]
    assert len(old_args["old_content"]) < 400 and len(old_args["new_content"]) < 400
    assert old_args["file_name"] == "/tmp/s/notes.md"
    assert len(new_args["old_content"]) == 2000 and len(new_args["new_content"]) == 2000


# ---------------------------------------------------------------------------
# ```ui block collapsing in older completed turns
# ---------------------------------------------------------------------------

UI_SELECT = "```ui\ntype: select\nprompt: 下一步？\noptions:\n  - 向左 [慢]\n  - 向右\n```"
UI_MULTI = "```ui\ntype: multiselect\nprompt: 带什么？\noptions: [水, 地图, 雨伞]\nmax: 2\n```"


def _ui_history(replies: list[str], block: str) -> list[Message]:
    """Turns: user -> assistant(block); the next user message answers it. Ends with an active user turn."""
    history: list[Message] = [_hmsg(MessageRole.USER, "开始", 0.0, "u_start")]
    for i, reply in enumerate(replies):
        history.append(_hmsg(MessageRole.ASSISTANT, f"正文{i}\n{block}", i * 100.0 + 10, f"a{i}"))
        history.append(_hmsg(MessageRole.USER, reply, i * 100.0 + 50, f"u{i}"))
    return history


def test_prepare_history_collapses_ui_select_to_plain_prompt() -> None:
    """An old select block collapses to its plain prompt text; the last completed turn keeps the full block."""
    texts = _assistant_texts(prepare_history_for_llm(_ui_history(["向左 [慢]", "向右", "随便"], UI_SELECT)))
    assert texts[0] == "正文0\n下一步？"
    assert texts[1] == "正文1\n下一步？"
    assert texts[2] == f"正文2\n{UI_SELECT}"


def test_prepare_history_collapses_ui_select_free_text_reply() -> None:
    """Regardless of user reply, only the plain prompt text is kept."""
    texts = _assistant_texts(prepare_history_for_llm(_ui_history(["我想想", "向右"], UI_SELECT)))
    assert texts[0] == "正文0\n下一步？"


def test_prepare_history_collapses_ui_multiselect_to_plain_prompt() -> None:
    """An old multiselect block collapses to its plain prompt text (options stripped)."""
    texts = _assistant_texts(prepare_history_for_llm(_ui_history(["选择：水；雨伞", "x"], UI_MULTI)))
    assert texts[0] == "正文0\n带什么？"
    assert "地图" not in texts[0]


def test_prepare_history_ui_collapse_deterministic_and_flag_off() -> None:
    """Collapsing is byte-stable across calls and disabled by strip_old_questions=False."""
    history = _ui_history(["向右", "向左 [慢]", "x"], UI_SELECT)
    a = _assistant_texts(prepare_history_for_llm(history))
    b = _assistant_texts(prepare_history_for_llm(_ui_history(["向右", "向左 [慢]", "x", "y"], UI_SELECT)))
    assert a[:2] == b[:2]
    off = _assistant_texts(prepare_history_for_llm(history, strip_old_questions=False))
    assert all(UI_SELECT in t for t in off)


def test_prepare_history_mixed_legacy_and_ui() -> None:
    """Legacy [question:] blocks keep their old placeholder behaviour next to ui blocks."""
    block = f"{Q_BLOCK}\n{UI_SELECT}"
    texts = _assistant_texts(prepare_history_for_llm(_ui_history(["向右", "x", "y"], block)))
    assert STRIPPED in texts[0]
    assert "下一步？" in texts[0]
    assert "```ui" not in texts[0]


# ---------------------------------------------------------------------------
# P0-1: head + tail history previews (shell truncation banner stripped)
# ---------------------------------------------------------------------------

B3_HEADER = (
    "=== STDOUT ===\nnetdiag v2.3 session diagnostic\nNEGOTIATED: codec=lz4-v2 max_frame=3072 token=a9f3c1\n"
    "--- packet dump (1200 packets) ---\n"
)
B3_BODY = "".join(f"pkt {i:05d} len=128 flags=ACK\n" for i in range(600))
B3_TAIL = "dropped=37 retransmits=4\ndiagnose complete (1200 packets)\n=== STDERR ===\n"


def _shell_banner(total: int, path: str) -> str:
    return (
        f"Output truncated (total length {total} bytes). Full output saved to session workspace file: `{path}`.\n"
        f'Page through it with `read_file("{path}", start_line=1)` (or re-run with a larger `max_output_chars`, '
        "max 30000).\n\nPreview (first ~45% and last ~55% of 5000 chars):\n"
    )


def _tool_turn(result: str | None = None, error: str | None = None) -> list[Message]:
    meta: dict = {"tool_name": "run_shell_command"}
    if error is not None:
        meta["tool_error"] = error
    else:
        meta["tool_result"] = result
    body = error if error is not None else result
    return [
        Message(id="u1", session_id="s", chatbot_id="cb", channel_id="ch", sender="u", role=MessageRole.USER,
                type=MessageType.TEXT, content="run it", status=MessageStatus.RESPONDED, timestamp=10),
        Message(id="r1", session_id="s", chatbot_id="cb", channel_id="ch", sender="run_shell_command",
                role=MessageRole.TOOL, type=MessageType.TOOL_RESULT, content=f"Tool returned:\n```\n{body}\n```",
                status=MessageStatus.RESPONDED, timestamp=11, metadata=meta),
        Message(id="u2", session_id="s", chatbot_id="cb", channel_id="ch", sender="u", role=MessageRole.USER,
                type=MessageType.TEXT, content="next", status=MessageStatus.RESPONDED, timestamp=20),
    ]


def test_strip_shell_truncation_banner_variants() -> None:
    from kesoku.agent.history import strip_shell_truncation_banner

    banner = _shell_banner(17000, "/s/cmd_output_1.txt")
    assert strip_shell_truncation_banner(banner + "BODY") == "BODY"
    failed = "Command failed with exit code 2.\n" + banner + "BODY"
    assert strip_shell_truncation_banner(failed) == "Command failed with exit code 2.\nBODY"
    legacy = "Output truncated (total length 9 bytes). Full output saved to x.\n\nPreview (last 5000 chars):\nBODY"
    assert strip_shell_truncation_banner(legacy) == "BODY"
    assert strip_shell_truncation_banner("plain output") == "plain output"
    assert strip_shell_truncation_banner("") == ""


def test_history_preview_keeps_b3_header_and_tail(tmp_path) -> None:
    """B3 shape: a ~140-char header, 16K of packet lines, a 2-line tail -> header and tail both in the preview."""
    full = B3_HEADER + B3_BODY + B3_TAIL
    saved = tmp_path / "cmd_output_1.txt"
    saved.write_text(full)
    shell_result = _shell_banner(len(full), str(saved)) + full[:2250] + "\n\n... <omitted> ...\n\n" + full[-2750:]
    prepared = prepare_history_for_llm(_tool_turn(result=shell_result), staging_dir=str(tmp_path))
    res = prepared[1].metadata["tool_result"]
    assert "NEGOTIATED: codec=lz4-v2 max_frame=3072 token=a9f3c1" in res
    assert "dropped=37 retransmits=4" in res
    assert "Preview (head and tail, 500 chars):" in res
    # S4: the shell's saved file is reused, no duplicate hist_* file is written
    assert prepared[1].metadata["output_filepath"] == str(saved)
    assert not list(tmp_path.glob("hist_*"))
    preview = res.split("Preview (head and tail, 500 chars):\n", 1)[1]
    assert len(preview) <= 500


def test_history_preview_error_variant_keeps_exit_code(tmp_path) -> None:
    full = B3_HEADER + B3_BODY + B3_TAIL
    err = "Command failed with exit code 2.\n" + _shell_banner(len(full), "/nonexistent/cmd_output_9.txt") + full
    prepared = prepare_history_for_llm(_tool_turn(error=err), staging_dir=str(tmp_path))
    out = prepared[1].metadata["tool_error"]
    assert "Error truncated for history" in out
    preview = out.split("Preview (head and tail, 500 chars):\n", 1)[1]
    assert preview.startswith("Command failed with exit code 2.\n=== STDOUT ===")
    assert "NEGOTIATED: codec=lz4-v2" in preview
    assert "diagnose complete (1200 packets)" in preview
    # deterministic err_* file written once per message id
    assert (tmp_path / "hist_err_run_shell_command_r1.txt").is_file() or list(tmp_path.glob("hist_err_*"))


def test_history_preview_short_outputs_untouched(tmp_path) -> None:
    prepared = prepare_history_for_llm(_tool_turn(result="short"), staging_dir=str(tmp_path))
    assert prepared[1].metadata["tool_result"] == "short"


def test_estimate_prepared_turn_tokens_uses_same_preview() -> None:
    from kesoku.agent.compressor import HistoryCompressor, estimate_tokens_calibrated
    from kesoku.agent.history import history_preview
    from kesoku.agent.llm import MockLLM

    full = B3_HEADER + B3_BODY + B3_TAIL
    turn = _tool_turn(result=full)[:2]
    comp = HistoryCompressor(db=None)  # type: ignore[arg-type]
    tokens = comp.estimate_prepared_turn_tokens(turn, MockLLM(), calibrated=True)
    expected = estimate_tokens_calibrated("run it") + estimate_tokens_calibrated(history_preview(full, 500))
    assert tokens == expected
