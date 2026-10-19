# kesoku-bench (Engineering Evaluation Suite)

`kesoku-bench` measures Kesoku as a coding agent end to end: 16 tasks in 4 suites (A multi-file
navigation & precision editing, B needle-in-the-head log debugging, C multi-turn compaction endurance,
D skills / background jobs / repo conventions), each graded by held-out pytest tests plus optional
behavioural checks. It is used to compare stages (baseline → P0 → P1 → P2) of the coding & compaction
upgrades on the same model.

Full specification: `kesoku_bench_spec.md` (Step 0 of the upgrade plan). This page is the operator guide.

## Layout

```
scripts/eval_bench.py                 # CLI: run | verify-gold | report | _run_one (child entry)
benchmarks/kesoku_bench/
  harness/
    taskspec.py   # task.toml schema (pydantic) + discovery/selection
    fixture.py    # per-task ws/ (AWD) + repo/ + out-of-tree holdout dir, config profiles, uv pre-warm
    metering.py   # Prices (from scripts/compression_sim.py), CostLedger, MeteredLLM proxy + breakers
    session.py    # BenchChatbot (headless adapter), BenchSession (Gateway + Agent wiring, idle, force_compaction)
    grading.py    # pytest/junit runner, tool-event extraction, tamper audit, grade.py integration
    runner.py     # run_task(): fixture -> turns -> grade -> TaskResult; run_suite(): child processes
    verify_gold.py# $0 fixture validation
    report.py     # scorecard (Markdown + JSON), stages side by side
    regrade.py    # $0 re-run of grade.py on saved results (after a grader fix), keeps <name>.json.orig
  common/         # config.template.toml (profiles), roles/eval_engineer, auto_reply.md, distractor_skills/
  tasks/<id>/     # task.toml, repo/, holdout/tests (+ref), gold/ (+apply.py), grade.py, skills/ (Suite D)
  results/        # gitignored: <stage>/<run_id>/<task_id>[.rN].json + scorecard.md/json
tests/benchmarks/ # harness unit tests (MockLLM end-to-end) + verify-gold gate (slow marker)
```

## How a task runs

1. `fixture.materialize` creates `/tmp/kesoku-bench/<run_id>/<task_id>/ws/` (the AWD: `config.toml`,
   `kesoku.db`, `sessions/`, `roles/eval_engineer/`, `skills/`, `repo/`). The repo is `uv sync`-ed and
   committed to a fresh git repo. Held-out tests are **not** on disk near the AWD during the run; the shipped
   visible tests are copied to a `kb-holdout-*` dir outside the work root for by-path restoration.
2. Each task runs in its own child process (kesoku keeps process-global config/AWD). `BenchSession` loads
   the rendered config (global AWD = `ws/`), wraps the model in `MeteredLLM`, patches `kesoku.agent.llm.get_llm`
   (so `/compact` is metered too), stubs embeddings and auto-rename, and starts `Gateway` + `Agent` +
   `BenchChatbot` in-process.
3. Scripted turns are posted as user messages; the harness waits for **session idle** (initiating message
   terminal, nothing `pending_agent`/`processing`, worker idle, stable for 1 s), so background-job wake-ups
   extend a turn. Questions without action get the canned auto-reply (max 2 per turn). `kind="command"`
   turns go through `Chatbot.execute_command_from_text`.
4. Suite C / D3 use `force_compact_before_turn`: `force_compaction()` runs the production watermark path with
   ~0 thresholds and `protect_tail_turns=1` so compaction points and chunk boundaries are identical across
   stages. Graded runs use profile `compaction_forced` (64K watermark, tail 2) so nothing else compacts.
5. Grading: tamper audit of tool-call arguments, restore shipped visible tests by path, copy held-out tests in,
   `uv run pytest --junitxml` on the declared F2P/P2P ids, then `grade.py` behavioural checks.
   `resolved = F2P ∧ P2P ∧ ¬tampered ∧ required behavioural checks`.

Timed-out or TLS-poisoned LLM calls (`Context has already been used to create a Connection`) are retried on a
freshly built SDK client (`MeteredLLM.inner_factory`), because the orphaned call keeps running in its thread.

Breakers: `--max-tool-iters 25` agent calls per scripted turn (proxy returns a stop response),
`--max-task-cost-usd 2.00`, `--max-suite-cost-usd 50.00` (a breach in a summarize call raises
`BudgetExceeded` so no bogus summary is persisted).

## `grade.py` contract

```python
def grade(ctx, gold_mode: bool = False) -> dict | GradeOutcome:
    # {"checks": {name: True | False | None}, "diagnostics": {...}}; None = skipped_in_gold_mode
def pre_cleanup(ctx) -> dict:  # optional; runs before the session and its background jobs are stopped
```

`ctx` (`GradeContext`): `task_id, task_dir, repo_dir, gold_mode, ws_dir, staging_dir, session_id, db_path,
turns, tool_events, messages, summary_nodes, llm_logs, llm_log_turn, junit, extras` plus `ctx.run(cmd)` and
`ctx.read(relpath)`. `extras` carries `port`, `forced_compaction_turns`, `active_jobs_for_session`,
`pre_cleanup`, and for command turns `manual_compact_nodes_delta`, `manual_compact_reply`, `fallback_forced`.

## Commands

```bash
uv run python scripts/eval_bench.py verify-gold                        # $0: all tasks
uv run python scripts/eval_bench.py verify-gold --tasks A1,B3 --keep   # keep throw-away fixtures
uv run python scripts/eval_bench.py run --tasks A1,B3,C2,D3 --stage baseline
uv run python scripts/eval_bench.py run --stage baseline --recommended-repeats --jobs 2
uv run python scripts/eval_bench.py run --suites C --natural-compaction   # diagnostic pass, never graded
uv run python scripts/eval_bench.py report --stage baseline --stage p0 --out /tmp/scorecard
uv run python scripts/eval_bench.py regrade --results benchmarks/kesoku_bench/results/baseline/<run_id> --tasks C3
uv run pytest tests/benchmarks/            # harness tests + verify-gold gate (-m "not slow" to skip it)
```

`verify-gold` checks per task: visible tests pass; every F2P id fails and every P2P id passes before the
fix; after overlaying `gold/` (and running `gold/apply.py <repo>` if present) and restoring the shipped tests,
the whole test directory passes in < 10 s of test time; `grade(ctx, gold_mode=True)` reports no `False` check.

## Writing a task

`task.toml`:

```toml
[task]
id = "C3_multi_file_migration_resume"   # must equal the directory name
suite = "C"
title = "..."
config_profile = "compaction_forced"   # default | compaction_forced | compaction_small
config_overrides = { "shell.background_threshold_seconds" = 8.0 }   # optional dotted keys
force_compact_before_turn = [5, 7]     # each in [4, n_turns]
critical_turn = 3                      # must be <= first forced point - 2
turn_timeout_s = 900
continue_on_timeout = false
fallback_force_if_noop = false         # C4: force_compaction(tail=2) if a /compact command made no node

[[turns]]
kind = "user"                          # user | command
text = """... {PORT} ... {AWD} ..."""

[grading]
fail_to_pass = ["tests/test_holdout_x.py::test_y"]   # files live in holdout/tests/
pass_to_pass = ["tests/test_loader.py"]              # shipped visible tests
behavioral = "grade.py"                              # optional
required_behavioral = ["plan_violations_zero"]       # gate `resolved`
```

Task repos are pure stdlib Python ≥ 3.12 with `[dependency-groups] dev = ["pytest>=8"]`, a `.gitignore`
covering `.venv/ __pycache__/ .diag_* server.pid requests.log`, ideally a committed `uv.lock`, and no hint of
the bug location.
