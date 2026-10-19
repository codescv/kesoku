---
name: proto-lint
description: Lint Protobuf (.proto) v2/v3 IDL schema definitions for field numbering, snake_case field names, and package declarations.
version: "1.0.0"
required_permissions:
  - run_shell_command
metadata:
  tags:
    - protobuf
    - proto
    - lint
    - idl
---

# Proto-Lint (`.proto` IDL Schema Linter)

This skill provides a static style and numbering linter for text-based Protocol Buffers (`.proto` v2 and v3) schema files.

## Usage

```bash
uv run python $SKILL_DIR/scripts/protolint.py <path/to/schema.proto> [--strict]
```

## Rules Enforced

1. File must declare `syntax = "proto3";` or `syntax = "proto2";` on the first non-comment line.
2. Package declaration (`package <org>.<service>;`) is required.
3. Message names must use `UpperCamelCase`; field names must use `lower_snake_case`.
4. Field tag numbers must be positive integers and must not fall in the reserved range `19000..19999`.

> [!NOTE]
> `proto-lint` only parses text `.proto` IDL files. It does not handle binary telemetry capture files.
