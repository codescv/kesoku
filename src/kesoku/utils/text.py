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
