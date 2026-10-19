---
name: i18n-extract
description: "Extract translatable strings from Python and Jinja sources into a gettext .pot catalogue."
version: "0.4.1"
---

# i18n-extract

Collects strings wrapped in `_()` / `gettext()` / `{% trans %}` and writes a `.pot` template for
translators.

## Usage

```bash
uv run python $SKILL_DIR/scripts/extract.py src/ templates/ --out locale/messages.pot
```

- `--keyword ngettext:1,2` adds plural keywords.
- Existing `.po` files are not touched; run `msgmerge` afterwards to update them.

Commit the `.pot` file together with the code change that added the strings.
