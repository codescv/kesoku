"""Unit tests for `kesoku.agent.reply_channel` (opt-in reply channel split and stream gate)."""

import pytest

from kesoku.agent.reply_channel import ReplySplit, ReplyStreamGate, split_reply
from kesoku.constants import REPLY_MARKER

M = REPLY_MARKER


def test_split_reply_basic() -> None:
    text = f"Private plan: check the list first.\n{M}\nHere is the answer."
    assert split_reply(text, M) == ReplySplit("Private plan: check the list first.", "Here is the answer.", True)


def test_split_reply_marker_first_line_has_empty_pre() -> None:
    assert split_reply(f"{M}\nHello there.\nSecond line.", M) == ReplySplit("", "Hello there.\nSecond line.", True)


def test_split_reply_absent_is_fail_open() -> None:
    text = "Just a normal reply without any marker."
    assert split_reply(text, M) == ReplySplit("", text, False)


def test_split_reply_empty_text() -> None:
    assert split_reply("", M) == ReplySplit("", "", False)


def test_split_reply_repeated_marker_lines_are_removed() -> None:
    text = f"plan A\n{M}\nPart one.\n{M}\nPart two."
    res = split_reply(text, M)
    assert res.found is True
    assert res.pre == "plan A"
    assert res.reply == "Part one.\nPart two."
    assert M not in res.reply


def test_split_reply_line_merely_containing_marker_is_not_a_split() -> None:
    text = f"The token {M} appears inline here.\nNothing else."
    assert split_reply(text, M) == ReplySplit("", text, False)


def test_split_reply_inline_marker_after_real_marker_is_kept() -> None:
    text = f"plan\n{M}\nUse {M} literally in a sentence."
    res = split_reply(text, M)
    assert res.found is True
    assert res.reply == f"Use {M} literally in a sentence."


def test_split_reply_tolerates_surrounding_whitespace_and_crlf() -> None:
    text = f"plan\r\n  {M}  \r\nReply body."
    res = split_reply(text, M)
    assert res.found is True
    assert res.pre == "plan"
    assert res.reply == "Reply body."


def test_split_reply_empty_reply_after_marker() -> None:
    assert split_reply(f"plan only\n{M}\n", M) == ReplySplit("plan only", "", True)


def test_gate_returns_none_until_marker_line_complete() -> None:
    gate = ReplyStreamGate(M)
    assert gate.feed("Thinking about the") is None
    assert gate.feed("Thinking about the answer.\n") is None
    assert gate.feed(f"Thinking about the answer.\n{M}") is None  # marker line not terminated yet
    assert gate.opened is False
    assert gate.feed(f"Thinking about the answer.\n{M}\n") == ""
    assert gate.opened is True
    assert gate.feed(f"Thinking about the answer.\n{M}\nHel") == "Hel"
    assert gate.feed(f"Thinking about the answer.\n{M}\nHello world") == "Hello world"


@pytest.mark.parametrize("cut", list(range(1, len(M))))
def test_gate_marker_split_across_chunk_boundaries(cut: int) -> None:
    gate = ReplyStreamGate(M)
    chunks = ["plan text\n", M[:cut], M[cut:], "\n", "Visible ", "reply."]
    accumulated = ""
    outputs: list[str | None] = []
    for c in chunks:
        accumulated += c
        outputs.append(gate.feed(accumulated))
    assert outputs[:3] == [None, None, None]
    assert outputs[-1] == "Visible reply."
    assert all("plan" not in o for o in outputs if o is not None)


def test_gate_line_containing_marker_with_other_text_never_opens() -> None:
    gate = ReplyStreamGate(M)
    assert gate.feed(f"prefix {M} suffix\nmore text\n") is None
    assert gate.opened is False


def test_gate_hides_trailing_partial_second_marker() -> None:
    gate = ReplyStreamGate(M)
    assert gate.feed(f"plan\n{M}\nPart one.\n{M[:4]}") == "Part one."
    assert gate.feed(f"plan\n{M}\nPart one.\n{M}\nPart two.") == "Part one.\nPart two."


def test_lenient_gate_passes_text_through_without_marker() -> None:
    gate = ReplyStreamGate(M, strict=False)
    assert gate.feed("Hello") == "Hello"
    assert gate.feed("Hello there.\nSecond") == "Hello there.\nSecond"
    assert gate.opened is False


def test_lenient_gate_hides_stray_marker_and_switches_to_reply() -> None:
    gate = ReplyStreamGate(M, strict=False)
    assert gate.feed("plan text\n") == "plan text\n"
    assert gate.feed(f"plan text\n{M[:5]}") == "plan text"  # partial marker held back
    assert gate.feed(f"plan text\n{M}") == "plan text"  # marker line not complete yet
    assert gate.feed(f"plan text\n{M}\nVisible") == "Visible"
    assert gate.opened is True
    assert gate.feed(f"plan text\n{M}\nVisible reply.") == "Visible reply."
