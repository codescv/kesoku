"""Unit tests for CJK/Japanese/ASCII FTS5 tokenization and wrapper stripping (`fts_tokenize`)."""

from kesoku.utils.text import fts_tokenize, strip_furigana, strip_injected_blocks


def test_cjk_bigrams_plus_trailing_unigram_for_index() -> None:
    """Verify overlapping bigrams plus trailing unigram `c_{n-1}` for CJK runs >= 2 (`C3`, `R2-3`)."""
    assert fts_tokenize("养了一只黑猫", for_query=False) == "养了 了一 一只 只黑 黑猫 猫"
    assert fts_tokenize("住新宿", for_query=False) == "住新 新宿 宿"
    assert fts_tokenize("东京出差", for_query=False) == "东京 京出 出差 差"


def test_isolated_single_cjk_unigram_and_ascii_lowering() -> None:
    """Verify isolated single CJK chars emit a unigram and ASCII words are lower-cased."""
    assert fts_tokenize("养 猫 PyTest_3", for_query=False) == "养 猫 pytest_3"


def test_furigana_stripping_appends_contiguous_kanji_copy_on_index() -> None:
    """Verify Japanese furigana `漢字(かな)` indexes both kana tokens and contiguous Kanji bigrams (`C3`)."""
    assert strip_furigana("新宿(しんじゅく)と渋谷（しぶや）") == "新宿と渋谷"
    indexed = fts_tokenize("住新宿(しんじゅく)", for_query=False)
    tokens = indexed.split()
    # Kana reading bigrams are preserved AND contiguous Kanji bigrams from stripped copy are appended
    assert "しん" in tokens
    assert "住新" in tokens
    assert "新宿" in tokens
    assert "宿" in tokens


def test_query_mode_adjacent_bigram_phrases_and_single_char_prefix() -> None:
    """Verify `for_query=True` emits FTS5 bigram phrases for runs >= 2 and `c*` prefix for single CJK (`C3`, `R2-3`)."""
    assert fts_tokenize("东京出差", for_query=True) == '"东京 京出 出差"'
    assert fts_tokenize("新宿", for_query=True) == '"新宿"'
    assert fts_tokenize("猫", for_query=True) == "猫*"
    assert fts_tokenize("新宿(しんじゅく) Ramen", for_query=True) == '"新宿" ramen'


def test_query_mode_escapes_fts5_keywords_and_punctuation_safely() -> None:
    """Verify reserved FTS5 operators (`NOT`, `AND`, `OR`, `NEAR`) and raw quotes/parens never break MATCH."""
    q = fts_tokenize('NOT "quoted" (near) and/or Python!', for_query=True)
    assert q == '"not" quoted "near" "and" "or" python'


def test_strip_injected_blocks_removes_all_xml_and_ui_wrappers() -> None:
    """Verify `<instructions>`, `<memory_pinned>`, `<recent_sessions>`, `<context>`, `<memory>`, and UI blocks strip."""
    raw = (
        "<instructions>\nPersona rules\n</instructions>\n"
        '<memory_pinned role="coder">\n- Prefers dark mode\n</memory_pinned>\n'
        '<recent_sessions role="coder">\n- #session:12345678 "Prev"\n</recent_sessions>\n'
        '<context name="notes">\nSession scratchpad\n</context>\n'
        "<memory>\n- $MEMORY_DIR/python.md\n</memory>\n"
        '<current_message from="User" time="2026-11-11">\n'
        "去新宿吃拉面吧\n"
        "```ui\ntype: select\noptions:\n  - 豚骨\n```\n"
        "[question: 选哪个？ || 豚骨 | 酱油]\n"
        "</current_message>"
    )
    assert strip_injected_blocks(raw) == "去新宿吃拉面吧"
    assert fts_tokenize(raw, for_query=False) == "去新 新宿 宿吃 吃拉 拉面 面吧 吧"
