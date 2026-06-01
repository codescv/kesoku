"""Text and markdown processing utility functions for Kesoku AI Agent."""

import re


def format_text(text: str) -> str:
    """Format/normalize markdown or lines before chunking.

    Cleans up headers, shifts header levels starting from level 1, clamps to
    maximum level 3, ensures blank line before headings, and collapses 3+
    consecutive newlines (outside code blocks).

    Args:
        text: The raw input markdown/text.

    Returns:
        The formatted and cleaned text.
    """
    if not text:
        return text

    # Split into lines preserving line endings
    lines = text.splitlines(keepends=True)

    # Pass 1: Identify all heading levels outside code blocks.
    in_code_block = False
    header_pattern = re.compile(r"^(#{1,6})\s+(.*)$")
    levels = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        match = header_pattern.match(line)
        if match:
            level = len(match.group(1))
            levels.append(level)

    # Determine shift amount to make the highest header level start at 1
    shift = 0
    if levels:
        min_level = min(levels)
        if min_level > 1:
            shift = min_level - 1

    # Pass 2: Rewrite headers with shift/clamp, collapse newlines and ensure spacing
    in_code_block = False
    formatted_lines = []
    consecutive_empty = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            formatted_lines.append(line)
            consecutive_empty = 0
            continue

        if in_code_block:
            formatted_lines.append(line)
            continue

        # Collapse consecutive empty lines to at most one empty line (two newlines)
        if not stripped:
            consecutive_empty += 1
            if consecutive_empty < 2:
                formatted_lines.append(line)
            continue
        else:
            consecutive_empty = 0

        # Process headers outside code blocks
        match = header_pattern.match(line)
        if match:
            orig_level = len(match.group(1))
            new_level = orig_level - shift
            if new_level < 1:
                new_level = 1
            if new_level > 3:
                new_level = 3

            header_text = match.group(2)
            line_ending = "\n"
            if line.endswith("\r\n"):
                line_ending = "\r\n"

            # Optimization: Ensure exactly one blank line before a header if not at the start
            if formatted_lines and not formatted_lines[-1].strip():
                # Previous line is already empty, so spacing is correct
                pass
            elif formatted_lines:
                # Previous line is not empty, insert a blank line
                formatted_lines.append(line_ending)

            formatted_lines.append("#" * new_level + " " + header_text + line_ending)
        else:
            formatted_lines.append(line)

    return "".join(formatted_lines)


def _close_chunk(chunk: str) -> str:
    """Append closing backticks to the chunk, ensuring proper newline.

    Args:
        chunk: The chunk text to close.

    Returns:
        The chunk closed with triple backticks.
    """
    if chunk.endswith("\n"):
        return chunk + "```"
    return chunk + "\n```"


def split_text_into_chunks(text: str, max_length: int) -> list[str]:
    """Split text into chunks of at most max_length.

    Avoids splitting in the middle of code blocks (triple backticks). If a
    chunk would exceed max_length, it closes the code block with triple
    backticks at the end of the current chunk, and prepends the matching
    opening tag at the beginning of the next chunk.

    Args:
        text: The formatted text to split.
        max_length: The maximum characters allowed in a single chunk.

    Returns:
        A list of message chunks.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    lines = text.splitlines(keepends=True)
    current_chunk = ""
    in_code_block = False
    code_block_header = ""

    for line in lines:
        stripped = line.strip()

        # Check if this line starts or ends a code block
        is_code_block_toggle = stripped.startswith("```")

        next_in_code_block = in_code_block
        next_code_block_header = code_block_header
        if is_code_block_toggle:
            if not in_code_block:
                next_in_code_block = True
                next_code_block_header = stripped
            else:
                next_in_code_block = False
                next_code_block_header = ""

        # Estimate total length of the current chunk with this line
        len_added = len(line)
        if next_in_code_block:
            # Account for closing backticks at the end of the chunk if we split after this line
            close_tag_len = 3 if line.endswith("\n") else 4
            total_len_with_line = len(current_chunk) + len_added + close_tag_len
        else:
            total_len_with_line = len(current_chunk) + len_added

        # Handle extremely long lines that exceed max_length
        if len_added > max_length or (in_code_block and len_added + len(code_block_header) + 5 > max_length):
            # Flush current chunk
            if current_chunk:
                if in_code_block:
                    chunks.append(_close_chunk(current_chunk))
                else:
                    chunks.append(current_chunk)
                current_chunk = ""

            # Force split the extremely long line
            limit = max_length
            if in_code_block:
                limit = max_length - len(code_block_header) - 5  # Safe margin
                if limit <= 0:
                    limit = max_length // 2

            for i in range(0, len(line), limit):
                sub_part = line[i : i + limit]
                if in_code_block:
                    sub_chunk = code_block_header + "\n" + sub_part
                    chunks.append(_close_chunk(sub_chunk))
                else:
                    chunks.append(sub_part)

            continue

        # Standard case: split if the line doesn't fit in the current chunk
        if total_len_with_line > max_length:
            if current_chunk:
                if in_code_block:
                    chunks.append(_close_chunk(current_chunk))
                else:
                    chunks.append(current_chunk)

            # Start new chunk with the line (and opening tag if currently in code block)
            if in_code_block:
                current_chunk = code_block_header + "\n" + line
            else:
                current_chunk = line
        else:
            # Append the line to the current chunk
            if not current_chunk and in_code_block:
                current_chunk = code_block_header + "\n" + line
            else:
                current_chunk += line

        in_code_block = next_in_code_block
        code_block_header = next_code_block_header

    if current_chunk:
        if in_code_block:
            chunks.append(_close_chunk(current_chunk))
        else:
            chunks.append(current_chunk)

    return chunks


def truncate_context_middle(text: str, max_len: int = 3000) -> str:
    """Truncates text in the middle if it exceeds max_len, preserving beginning and end.

    It attempts to split cleanly on newline boundaries to preserve markdown list formatting.

    Args:
        text: The input context string.
        max_len: Maximum allowed character length.

    Returns:
        The truncated string with a deletion indicator in the middle.
    """
    if len(text) <= max_len:
        return text

    # Split into 40% start, 55% end, keeping 5% buffer for the indicator
    keep_start = int(max_len * 0.4)
    keep_end = int(max_len * 0.55)

    # Try to find clean newline boundaries near the split indices (+-100 chars)
    # to prevent breaking markdown list items/lines in half
    start_idx = text.find("\n", max(0, keep_start - 100), min(len(text), keep_start + 100))
    if start_idx == -1:
        start_idx = keep_start

    end_idx = text.rfind("\n", max(0, len(text) - keep_end - 100), min(len(text), len(text) - keep_end + 100))
    if end_idx == -1:
        end_idx = len(text) - keep_end

    if start_idx < end_idx:
        return text[:start_idx] + "\n\n... [Timeline Truncated for Brevity] ...\n\n" + text[end_idx:]

    # Fallback to simple raw slice if boundary matching fails or is overlapping
    char_start = int(max_len * 0.45)
    char_end = int(max_len * 0.45)
    return text[:char_start] + "\n\n... [Timeline Truncated for Brevity] ...\n\n" + text[-char_end:]


# LaTeX to Unicode symbol mapping for Discord readability
LATEX_SYMBOL_MAP = {
    # Greek letters (lowercase)
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\epsilon": "ε",
    r"\zeta": "ζ",
    r"\eta": "η",
    r"\theta": "θ",
    r"\iota": "ι",
    r"\kappa": "κ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\nu": "ν",
    r"\xi": "ξ",
    r"\pi": "π",
    r"\rho": "ρ",
    r"\sigma": "σ",
    r"\tau": "τ",
    r"\upsilon": "υ",
    r"\phi": "φ",
    r"\chi": "χ",
    r"\psi": "ψ",
    r"\omega": "ω",
    # Greek letters (uppercase)
    r"\Gamma": "Γ",
    r"\Delta": "Δ",
    r"\Theta": "Θ",
    r"\Lambda": "Λ",
    r"\Xi": "Ξ",
    r"\Pi": "Π",
    r"\Sigma": "Σ",
    r"\Upsilon": "Υ",
    r"\Phi": "Φ",
    r"\Psi": "Ψ",
    r"\Omega": "Ω",
    # Operators & Relations
    r"\sum": "∑",
    r"\prod": "∏",
    r"\int": "∫",
    r"\approx": "≈",
    r"\neq": "≠",
    r"\le": "≤",
    r"\leq": "≤",
    r"\ge": "≥",
    r"\geq": "≥",
    r"\times": "×",
    r"\cdot": "·",
    r"\div": "÷",
    r"\pm": "±",
    r"\infty": "∞",
    r"\to": "→",
    r"\rightarrow": "→",
    r"\leftarrow": "←",
    r"\Rightarrow": "⇒",
    r"\Leftarrow": "⇐",
    r"\leftrightarrow": "↔",
    r"\Leftrightarrow": "⇔",
    r"\in": "∈",
    r"\notin": "∉",
    r"\subset": "⊂",
    r"\supset": "⊃",
    r"\subseteq": "⊆",
    r"\supseteq": "⊇",
    r"\cap": "∩",
    r"\cup": "∪",
    r"\forall": "∀",
    r"\exists": "∃",
    r"\nabla": "∇",
    r"\partial": "∂",
    r"\empty": "∅",
    r"\emptyset": "∅",
    r"\sqrt": "√",
}

SUPERSCRIPT_MAP = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
    "n": "ⁿ", "i": "ⁱ", "x": "ˣ",
}

SUBSCRIPT_MAP = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
    "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
    "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
    "v": "ᵥ", "x": "ₓ",
}


def _to_superscript(text: str) -> str | None:
    res = []
    for c in text:
        if c in SUPERSCRIPT_MAP:
            res.append(SUPERSCRIPT_MAP[c])
        else:
            return None
    return "".join(res)


def _to_subscript(text: str) -> str | None:
    res = []
    for c in text:
        if c in SUBSCRIPT_MAP:
            res.append(SUBSCRIPT_MAP[c])
        else:
            return None
    return "".join(res)


def _clean_latex_expression(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return ""

    # 1. Fractions: \frac{num}{den} -> (num)/(den)
    prev = ""
    while prev != expr:
        prev = expr
        expr = re.sub(r'\\frac\s*{(.*?)}\s*{(.*?)}', r'(\1)/(\2)', expr)

    # 2. Square roots: \sqrt{expr} -> √(expr), \sqrt[n]{expr} -> ⁿ√(expr)
    prev = ""
    while prev != expr:
        prev = expr
        expr = re.sub(
            r'\\sqrt\s*\[(.*?)\]\s*{(.*?)}',
            lambda m: f"{_to_superscript(m.group(1)) or f'({m.group(1)})'}√({m.group(2)})",
            expr,
        )
        expr = re.sub(r'\\sqrt\s*{(.*?)}', r'√(\1)', expr)

    # 3. Superscripts and Subscripts
    expr = re.sub(
        r'\^{(.*?)}',
        lambda m: _to_superscript(content := m.group(1)) or f"^({content})",
        expr,
    )
    expr = re.sub(
        r'_{(.*?)}',
        lambda m: _to_subscript(content := m.group(1)) or f"_({content})",
        expr,
    )

    expr = re.sub(
        r'\^([a-zA-Z0-9+-=])',
        lambda m: _to_superscript(m.group(1)) or f"^({m.group(1)})",
        expr,
    )
    expr = re.sub(
        r'_([a-zA-Z0-9+-=])',
        lambda m: _to_subscript(m.group(1)) or f"_({m.group(1)})",
        expr,
    )

    # 4. Replace LaTeX symbols
    sorted_symbols = sorted(LATEX_SYMBOL_MAP.keys(), key=len, reverse=True)
    for sym in sorted_symbols:
        expr = expr.replace(sym, LATEX_SYMBOL_MAP[sym])

    # 5. Remove common LaTeX formatting commands
    expr = re.sub(r'\\mathbf\s*{(.*?)}', r'**\1**', expr)
    expr = re.sub(r'\\mathit\s*{(.*?)}', r'*\1*', expr)
    expr = re.sub(r'\\mathrm\s*{(.*?)}', r'\1', expr)
    expr = re.sub(r'\\text\s*{(.*?)}', r'\1', expr)
    expr = re.sub(r'\\label\s*{(.*?)}', '', expr)

    # 6. Clean up remaining backslashes
    expr = re.sub(r'\\([\,>\s])', r'\1', expr)
    expr = re.sub(r'\\([{}])', r'\1', expr)

    # Remove environments
    expr = re.sub(r'\\begin{[a-zA-Z]*?}', '', expr)
    expr = re.sub(r'\\end{[a-zA-Z]*?}', '', expr)

    expr = re.sub(r'\s+', ' ', expr)

    return expr.strip()


def _clean_latex_inline(expr: str) -> str:
    return _clean_latex_expression(expr)


def _clean_latex_block(expr: str) -> str:
    cleaned = _clean_latex_expression(expr)
    lines = [
        f"> {line.strip()}"
        for line in cleaned.splitlines()
        if line.strip()
    ]
    if not lines:
        return ""
    return "\n" + "\n".join(lines) + "\n"


def _clean_latex_outside_code_block(text: str) -> str:
    # Replace block math \[...\]
    text = re.sub(
        r'\\\[(.*?)\\\]',
        lambda m: _clean_latex_block(m.group(1)),
        text,
        flags=re.DOTALL,
    )

    # Replace block math $$...$$
    text = re.sub(
        r'\$\$(.*?)\$\$',
        lambda m: _clean_latex_block(m.group(1)),
        text,
        flags=re.DOTALL,
    )

    # Replace inline math \(...\)
    text = re.sub(
        r'\\\((.*?)\\\)',
        lambda m: _clean_latex_inline(m.group(1)),
        text,
    )

    # Replace inline math $...$
    text = re.sub(
        r'\$(?!\s)(.*?)(?<!\s)\$',
        lambda m: _clean_latex_inline(m.group(1)),
        text,
    )

    return text


def clean_latex(text: str) -> str:
    """Replace LaTeX formulas with plain text or rich text symbols for readability.

    Args:
        text: The input text containing LaTeX formulas.

    Returns:
        Text with LaTeX formulas replaced by readable unicode representations.
    """
    if not text:
        return text

    parts = text.split("```")
    for i in range(len(parts)):
        if i % 2 == 0:  # Even index means outside code block
            parts[i] = _clean_latex_outside_code_block(parts[i])

    return "```".join(parts)


