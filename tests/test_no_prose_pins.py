"""Meta-test guard: prevent exact prompt line-count locks and prose-regex pins in `tests/**/*.py`."""

import ast
import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SELF_PATH = Path(__file__).resolve()
PROMPT_CONSTANTS = {"SESSION_NOTES_INSTRUCTIONS", "OUTPUT_FORMATTING_INSTRUCTIONS"}
PROMPT_TARGET_RE = re.compile(r"prompt|intro|pref|instruct|skill|game|\.md\b", re.IGNORECASE)
LINE_COUNT_LOCK_RE = re.compile(r"\bsplitlines\(\)\)\s*==")


def _scan_test_source(rel: str, text: str) -> list[str]:
    """Return prompt line-count or prose-regex violations found in `text`."""
    hits: list[str] = []
    tree = ast.parse(text, filename=rel)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        fn_src = ast.get_source_segment(text, node) or ""
        for offset, line in enumerate(fn_src.splitlines()):
            if LINE_COUNT_LOCK_RE.search(line) and PROMPT_TARGET_RE.search(fn_src):
                hits.append(f"{rel}:{node.lineno + offset}: exact prompt line-count lock (`splitlines()) ==`)")
        if any(name in fn_src for name in PROMPT_CONSTANTS) and re.search(r"\bre\.(search|match|findall)\s*\(", fn_src):
            hits.append(f"{rel}:{node.lineno}: prose-regex lock in `{node.name}` over prompt instructions")
    return hits


def test_no_prompt_line_count_or_prose_regex_pins() -> None:
    """No test under `tests/**/*.py` pins `splitlines()) ==` on prompts or regex-locks prompt prose."""
    violations: list[str] = []
    for path in sorted(TESTS_DIR.rglob("*.py")):
        if path.resolve() == SELF_PATH or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(TESTS_DIR.parent).as_posix()
        violations.extend(_scan_test_source(rel, path.read_text(encoding="utf-8")))
    assert not violations, "prose/line-count pins detected:\n" + "\n".join(violations)


def test_prose_pin_scanner_catches_line_lock_and_regex_pin() -> None:
    """Verify `_scan_test_source` flags prompt `splitlines()) ==` and `re.search` on prompt constants."""
    bad = (
        "def test_bad():\n"
        "    assert len(prompt.splitlines()) == 75\n"
        "    assert re.search(r'never edit', SESSION_NOTES_INSTRUCTIONS)\n"
    )
    assert len(_scan_test_source("fake.py", bad)) == 2
