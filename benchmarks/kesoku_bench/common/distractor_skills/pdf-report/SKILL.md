---
name: pdf-report
description: "Render a Markdown or HTML report into a branded PDF with cover page and table of contents."
version: "1.3.0"
---

# pdf-report

Turns a Markdown/HTML report into a PDF using the team template (cover page, TOC, page numbers).

## Usage

```bash
uv run python $SKILL_DIR/scripts/render_pdf.py report.md --out report.pdf --title "Q3 Review"
```

- `--toc` adds a table of contents from `#`/`##` headings.
- `--landscape` for wide tables.

Notes: images are embedded at 150 dpi; keep source files under 20 MB. Use for stakeholder reports, not
for code documentation.
