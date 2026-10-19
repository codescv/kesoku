---
name: perf-flamegraph
description: "Profile a Python command with cProfile and render a collapsed-stack flame graph SVG."
version: "1.2.3"
---

# perf-flamegraph

Profiles a Python entry point and writes a flame graph so you can see where time goes.

## Usage

```bash
uv run python $SKILL_DIR/scripts/flame.py --out profile.svg -- -m mypkg.main --input big.json
```

- `--min-pct 0.5` hides frames below 0.5% of total time.
- `--collapsed out.txt` also writes the collapsed stacks (Brendan Gregg format).

Profiling adds ~2x overhead; compare relative widths, not absolute times.
