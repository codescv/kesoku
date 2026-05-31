"""Markdown table parser and high-quality CJK/Latin image renderer for Kesoku."""

import io
import re
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFont


class ParsedTable(NamedTuple):
    """Holds structural data and text range offsets for a parsed markdown table."""

    start_idx: int
    end_idx: int
    headers: list[str]
    alignments: list[str]  # "left", "center", "right"
    rows: list[list[str]]
    raw_text: str


class FallbackFont:
    """Font wrapper providing character-by-character fallback between Latin and CJK fonts."""

    def __init__(self, latin_font_path: str, cjk_font_path: str, size: int):
        """Initialize Latin and CJK TrueType fallback fonts.

        Args:
            latin_font_path: Absolute path to the Latin font.
            cjk_font_path: Absolute path to the CJK font.
            size: Font size in points.
        """
        try:
            self.latin_font = ImageFont.truetype(latin_font_path, size)
        except Exception:
            self.latin_font = ImageFont.load_default()

        try:
            self.cjk_font = ImageFont.truetype(cjk_font_path, size)
        except Exception:
            self.cjk_font = self.latin_font

        self.size = size

    def _get_font_for_char(self, char: str) -> ImageFont.FreeTypeFont:
        # Unicode block 0x2E80 starts CJK radicals supplement.
        # Anything below is typically Latin, numbers, punctuation, and symbols.
        if ord(char) < 0x2E80:
            return self.latin_font
        return self.cjk_font

    def getbbox(self, text: str) -> tuple[int, int, int, int]:
        """Calculate horizontal/vertical bounding box using correct character fonts."""
        if not text:
            return 0, 0, 0, self.size

        total_w = 0
        max_h = 0
        for char in text:
            font = self._get_font_for_char(char)
            bbox = font.getbbox(char)
            char_w = bbox[2] - bbox[0]
            char_h = bbox[3] - bbox[1]

            if char_w == 0:
                char_w = self.size // 2 if char == " " else 0

            total_w += char_w
            if char_h > max_h:
                max_h = char_h

        return 0, 0, total_w, max_h or self.size

    def draw_text(
        self,
        draw: ImageDraw.ImageDraw,
        position: tuple[int, int],
        text: str,
        fill: tuple[int, int, int],
    ):
        """Draw text character by character dynamically switching Latin/CJK fonts."""
        start_x, start_y = position
        curr_x = start_x
        for char in text:
            font = self._get_font_for_char(char)
            bbox = font.getbbox(char)
            char_w = bbox[2] - bbox[0]

            if char_w == 0:
                char_w = self.size // 2 if char == " " else 0

            draw.text((curr_x, start_y), char, font=font, fill=fill)
            curr_x += char_w


def split_cells(line: str) -> list[str]:
    """Split a table row by pipe, respecting escaped pipes."""
    placeholder = "___ESCAPED_PIPE___"
    escaped = line.replace(r"\|", placeholder)
    parts = [p.replace(placeholder, "|").strip() for p in escaped.split("|")]
    return parts


def parse_markdown_tables(text: str) -> list[ParsedTable]:
    """Parse all contiguous markdown tables within the given text.

    Args:
        text: The text containing markdown tables.

    Returns:
        A list of ParsedTable tuples detailing table structure and character ranges.
    """
    # A table divider row looks like: |:---|:---:|---:|
    # It must contain only spaces, colons, dashes, and pipes.
    divider_pattern = re.compile(r"^\|?(\s*:?-+:?\s*\|)*\s*:?-+:?\s*\|?\s*$")

    lines = text.splitlines()
    tables = []
    i = 0
    n = len(lines)

    # Calculate start/end character offsets for each line in the original text
    line_offsets = []
    current_offset = 0
    for line in lines:
        next_idx = text.find(line, current_offset)
        if next_idx != -1:
            current_offset = next_idx + len(line)
            line_offsets.append((next_idx, current_offset))
            # Advance past trailing line breaks in original text
            while current_offset < len(text) and text[current_offset] in ("\r", "\n"):
                current_offset += 1
        else:
            line_offsets.append((current_offset, current_offset + len(line)))
            current_offset += len(line) + 1

    while i < n - 1:
        line = lines[i].strip()
        next_line = lines[i + 1].strip()

        # A table requires headers (with pipes) and a following divider row
        if "|" in line and "|" in next_line and divider_pattern.match(next_line):
            header_line = lines[i]
            divider_line = lines[i + 1]

            # Extract and clean header cells
            headers = split_cells(header_line)
            if header_line.strip().startswith("|"):
                headers = headers[1:]
            if header_line.strip().endswith("|"):
                headers = headers[:-1]

            # Extract and resolve column alignments
            divider_cells = split_cells(divider_line)
            if divider_line.strip().startswith("|"):
                divider_cells = divider_cells[1:]
            if divider_line.strip().endswith("|"):
                divider_cells = divider_cells[:-1]

            alignments = []
            for cell in divider_cells:
                has_left = cell.startswith(":")
                has_right = cell.endswith(":")
                if has_left and has_right:
                    alignments.append("center")
                elif has_right:
                    alignments.append("right")
                else:
                    alignments.append("left")

            # Fallback/pad alignments to match headers
            while len(alignments) < len(headers):
                alignments.append("left")
            alignments = alignments[: len(headers)]

            # Parse following contiguous data rows
            rows = []
            j = i + 2
            while j < n and "|" in lines[j]:
                data_line = lines[j]
                if divider_pattern.match(data_line.strip()):
                    break
                cells = split_cells(data_line)
                if data_line.strip().startswith("|"):
                    cells = cells[1:]
                if data_line.strip().endswith("|"):
                    cells = cells[:-1]

                # Conform cells to headers length
                while len(cells) < len(headers):
                    cells.append("")
                cells = cells[: len(headers)]
                rows.append(cells)
                j += 1

            # Record exact start and end offsets in original text
            start_char = line_offsets[i][0]
            end_char = line_offsets[j - 1][1]
            raw_table_text = text[start_char:end_char]

            tables.append(
                ParsedTable(
                    start_idx=start_char,
                    end_idx=end_char,
                    headers=headers,
                    alignments=alignments,
                    rows=rows,
                    raw_text=raw_table_text,
                )
            )
            i = j  # Resume after parsed table
        else:
            i += 1

    return tables


def render_table_to_image(
    headers: list[str],
    alignments: list[str],
    rows: list[list[str]],
    latin_font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    cjk_font_path: str = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    font_size: int = 16,
) -> bytes:
    """Render a parsed markdown table into a beautifully styled PNG image with CJK support.

    Args:
        headers: List of column header strings.
        alignments: List of "left", "center", or "right" alignments per column.
        rows: List of rows, where each row is a list of cell strings.
        latin_font_path: Path to the TrueType Latin font.
        cjk_font_path: Path to the TrueType font supporting CJK characters.
        font_size: Base font size in points.

    Returns:
        Raw PNG file bytes.
    """
    # Clean up cell contents (e.g., replace HTML breaks or escaped breaks with newlines)
    headers = [h.replace("<br>", "\n").replace("<br/>", "\n").replace("\\n", "\n") for h in headers]
    rows = [[c.replace("<br>", "\n").replace("<br/>", "\n").replace("\\n", "\n") for c in r] for r in rows]

    # Load fallback fonts
    font = FallbackFont(latin_font_path, cjk_font_path, font_size)
    try:
        bold_font = FallbackFont(latin_font_path.replace(".ttf", "-Bold.ttf"), cjk_font_path, font_size)
    except Exception:
        bold_font = font

    # Padding & border config
    cell_padding_x = 16
    cell_padding_y = 12
    border_width = 1

    # Professional color scheme
    bg_color = (255, 255, 255)
    header_bg_color = (44, 62, 80)  # Slate grey-blue
    header_text_color = (255, 255, 255)
    cell_text_color = (33, 37, 41)
    border_color = (222, 226, 230)
    alt_row_bg_color = (248, 249, 250)

    num_cols = len(headers)
    num_rows = len(rows)

    def get_text_size(text: str, f: FallbackFont) -> tuple[int, int]:
        if not text:
            return 0, font_size
        lines = text.split("\n")
        max_w = 0
        total_h = 0
        for line_str in lines:
            bbox = f.getbbox(line_str)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            if w > max_w:
                max_w = w
            total_h += h + 4
        return max_w, total_h - 4

    # Initialize dimensions
    col_widths = [0] * num_cols
    row_heights = [0] * (num_rows + 1)

    # Measure column widths & row heights
    for c_idx, h_text in enumerate(headers):
        w, h = get_text_size(h_text, bold_font)
        col_widths[c_idx] = max(col_widths[c_idx], w + cell_padding_x * 2)
        row_heights[0] = max(row_heights[0], h + cell_padding_y * 2)

    for r_idx, row in enumerate(rows):
        for c_idx, cell_text in enumerate(row):
            w, h = get_text_size(cell_text, font)
            col_widths[c_idx] = max(col_widths[c_idx], w + cell_padding_x * 2)
            row_heights[r_idx + 1] = max(row_heights[r_idx + 1], h + cell_padding_y * 2)

    total_width = sum(col_widths) + border_width * (num_cols + 1)
    total_height = sum(row_heights) + border_width * (num_rows + 2)

    # Initialize canvas
    image = Image.new("RGB", (total_width, total_height), bg_color)
    draw = ImageDraw.Draw(image)

    # Draw header bg
    draw.rectangle(
        [(border_width, border_width), (total_width - border_width, border_width + row_heights[0])],
        fill=header_bg_color,
    )

    # Draw alternating row bgs
    current_y = border_width + row_heights[0] + border_width
    for r_idx in range(num_rows):
        row_h = row_heights[r_idx + 1]
        if r_idx % 2 == 1:
            draw.rectangle(
                [(border_width, current_y), (total_width - border_width, current_y + row_h)], fill=alt_row_bg_color
            )
        current_y += row_h + border_width

    # Draw borders
    # Horizontal
    current_y = 0
    draw.line([(0, current_y), (total_width, current_y)], fill=border_color, width=border_width)
    current_y += border_width + row_heights[0]
    draw.line([(0, current_y), (total_width, current_y)], fill=border_color, width=border_width)
    for r_h in row_heights[1:]:
        current_y += border_width + r_h
        draw.line([(0, current_y), (total_width, current_y)], fill=border_color, width=border_width)

    # Vertical
    current_x = 0
    draw.line([(current_x, 0), (current_x, total_height)], fill=border_color, width=border_width)
    for col_w in col_widths:
        current_x += border_width + col_w
        draw.line([(current_x, 0), (current_x, total_height)], fill=border_color, width=border_width)

    def draw_cell_text(x, y, cell_w, cell_h, text, f: FallbackFont, color, align):
        lines = text.split("\n")
        line_heights = []
        total_text_h = 0
        for line_str in lines:
            bbox = f.getbbox(line_str)
            h = bbox[3] - bbox[1]
            line_heights.append(h)
            total_text_h += h + 4
        total_text_h -= 4

        start_y = y + (cell_h - total_text_h) // 2
        curr_y = start_y

        for line_idx, line_str in enumerate(lines):
            if not line_str:
                curr_y += font_size + 4
                continue
            bbox = f.getbbox(line_str)
            w = bbox[2] - bbox[0]
            h = line_heights[line_idx]

            if align == "center":
                start_x = x + (cell_w - w) // 2
            elif align == "right":
                start_x = x + cell_w - w - cell_padding_x
            else:
                start_x = x + cell_padding_x

            f.draw_text(draw, (start_x, curr_y), line_str, fill=color)
            curr_y += h + 4

    # Draw header text
    current_x = border_width
    for c_idx, h_text in enumerate(headers):
        draw_cell_text(
            current_x,
            border_width,
            col_widths[c_idx],
            row_heights[0],
            h_text,
            bold_font,
            header_text_color,
            alignments[c_idx],
        )
        current_x += col_widths[c_idx] + border_width

    # Draw cell text
    current_y = border_width + row_heights[0] + border_width
    for r_idx, row in enumerate(rows):
        row_h = row_heights[r_idx + 1]
        current_x = border_width
        for c_idx, cell_text in enumerate(row):
            draw_cell_text(
                current_x,
                current_y,
                col_widths[c_idx],
                row_h,
                cell_text,
                font,
                cell_text_color,
                alignments[c_idx],
            )
            current_x += col_widths[c_idx] + border_width
        current_y += row_h + border_width

    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()
