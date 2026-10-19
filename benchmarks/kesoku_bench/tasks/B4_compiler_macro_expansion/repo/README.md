# `tmplc` — Typed Template & Macro Compiler

`tmplc` compiles `.tpl` template files and shared macro libraries (`_macros.tpl`) through a 4-stage pipeline:

1. **Lexer & Parser (`tmplc.lexer`, `tmplc.parser`)**: Parses macro declarations (`macro name(arg: Type = default, ...)`), static HTML/text blocks, and macro invocations (`@name(arg=val, ...)`).
2. **Macro Expansion (`tmplc.expand`)**: Resolves macro calls against declared `MacroDef` signatures, binds omitted arguments via `bind_defaults()`, and lowers templates into an Expanded Intermediate Representation (`ExpandedIR`).
3. **Type Checker (`tmplc.typecheck`)**: Validates every bound argument against the declared parameter type (`Int`, `Bool`, `Float`, `Str`) and provides `coerce_literal()` for converting raw literal tokens into `TypedValue` objects.
4. **Code Emitter (`tmplc.emit`)**: Renders verified `ExpandedIR` modules into HTML fragments.

## CLI

```bash
uv run python -m tmplc build templates/
uv run pytest
```
