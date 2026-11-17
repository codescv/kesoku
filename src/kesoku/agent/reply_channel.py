"""Opt-in reply channel: separate private pre-reply text from the user-visible reply.

When the reply channel is enabled for a turn, the model is instructed to start its final
user-visible reply with a line containing exactly the reply marker
(`kesoku.constants.REPLY_MARKER`). Anything written before that line is private: it is
persisted as a THOUGHT row and never streamed or delivered.

The split is purely structural (an exact marker line) and fails open: when the marker is
absent the whole text is treated as the reply, exactly as without the channel.
"""

from typing import NamedTuple


class ReplySplit(NamedTuple):
    """Result of splitting a final response at the reply marker."""

    pre: str
    """Private text that preceded the marker line (empty when the marker is absent)."""
    reply: str
    """User-visible reply with every marker line removed."""
    found: bool
    """Whether a marker line was present."""


def _is_marker_line(line: str, marker: str) -> bool:
    """Return True when `line` (without its newline) consists of exactly the marker."""
    return line.strip() == marker


def split_reply(text: str, marker: str) -> ReplySplit:
    """Split `text` at the first line equal to `marker`.

    - The text before the first marker line becomes `pre` (stripped).
    - The text after it becomes `reply`; any later marker lines are removed.
    - A line that merely contains the marker alongside other text is not a marker line.
    - No marker line -> `ReplySplit("", text, False)` (fail-open).

    Args:
        text: The full final response text.
        marker: The marker string.

    Returns:
        The `ReplySplit`.
    """
    if not text or not marker or marker not in text:
        return ReplySplit("", text, False)

    lines = text.splitlines(keepends=True)
    for idx, line in enumerate(lines):
        if _is_marker_line(line, marker):
            pre = "".join(lines[:idx]).strip()
            rest = [ln for ln in lines[idx + 1 :] if not _is_marker_line(ln, marker)]
            reply = "".join(rest).strip()
            return ReplySplit(pre, reply, True)
    return ReplySplit("", text, False)


class ReplyStreamGate:
    """Incremental gate for streamed previews of the accumulated response text.

    `feed` receives the *accumulated* text of the stream so far (not deltas), so a marker that
    is split across chunk boundaries is handled naturally.

    - strict (reply channel on for the turn): nothing passes until the marker line is seen.
    - lenient (reply channel off): text passes through, but a marker line the model emits anyway
      (e.g. imitated from history) is never shown: once it is complete, only the post-marker reply
      is previewed, and a trailing partial marker line is held back.
    """

    def __init__(self, marker: str, strict: bool = True) -> None:
        """Create a gate for `marker` (`strict=False` for turns where the channel is off)."""
        self.marker = marker
        self.strict = strict
        self._opened = False

    @property
    def opened(self) -> bool:
        """Whether the marker line has been seen."""
        return self._opened

    def _hide_partial_marker_tail(self, text: str) -> str:
        """Drop a trailing (still growing) line that could become a marker line."""
        last_nl = text.rfind("\n")
        tail = text[last_nl + 1 :].strip()
        if tail and self.marker.startswith(tail):
            return text[: last_nl + 1].rstrip() if last_nl >= 0 else ""
        return text

    def feed(self, accumulated: str) -> str | None:
        """Return the preview text for `accumulated`, or None while nothing may be shown.

        The marker line only counts once it is complete (terminated by a newline), so a partial
        line such as ``"<<<REP"`` or a line that continues past the marker never opens the gate.
        """
        if not self._opened:
            complete = accumulated.rpartition("\n")[0] if self.marker in accumulated else ""
            # Only complete lines can be judged: the last line may still be growing.
            if complete and any(_is_marker_line(ln, self.marker) for ln in complete.split("\n")):
                self._opened = True
            elif self.strict:
                return None
            else:
                return self._hide_partial_marker_tail(accumulated)
        result = split_reply(accumulated, self.marker)
        if not result.found:
            return None
        # A trailing partial marker line in the reply must not leak into the preview.
        return self._hide_partial_marker_tail(result.reply)
