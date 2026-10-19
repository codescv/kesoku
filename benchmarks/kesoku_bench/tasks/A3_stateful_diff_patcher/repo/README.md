# upatch

`upatch` is a pure-stdlib unified diff parser and stateful patch engine.

## Modules

- `upatch/parse.py`: Unified diff parser producing `FilePatch` and `Hunk` objects.
- `upatch/lineend.py`: Line-ending inspection, splitting, and reconstruction utilities (`LF` vs `CRLF`).
- `upatch/apply.py`: Stateful multi-hunk patch applicator supporting exact, offset, and fuzz matching modes.
- `upatch/io.py`: File reading and writing helpers for patching files on disk.
- `upatch/cli.py`: Command-line interface (`upatch apply <patch_file> <target_file> [-o <out_file>] [--in-place]`).

## Usage

Apply a patch and print the patched bytes to stdout (leaving `<target_file>` untouched unless `--in-place` or `-o` is passed):

```bash
uv run upatch apply fixtures/single.patch fixtures/single.txt
```

Write the patched output to a file:

```bash
uv run upatch apply fixtures/multi.patch fixtures/multi.txt -o /tmp/multi.out
```

Run tests:

```bash
uv run pytest
```
