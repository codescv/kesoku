import pytest

from kesoku.agent.llm import BaseLLM, LLMResponse, ToolCallRequest
from kesoku.agent.tools.registry import ToolContext
from kesoku.agent.tools.shell import run_shell_command
from kesoku.agent.tools.subagent import delegate_subagent
from kesoku.config import KesokuConfig
from kesoku.context import KesokuContext


class SequencedLLM(BaseLLM):
    """Deterministic LLM double returning canned LLMResponses in sequence."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.model_name = "mock-subagent"

    async def generate(
        self,
        prompt=None,
        system_prompt=None,
        history=None,
        tools=None,
        cached_content=None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "history": history,
                "tools": [getattr(t, "__name__", str(t)) for t in (tools or [])],
            }
        )
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_subagent_tool_rejection():
    ctx = ToolContext(session_id="s1", is_subagent=True)
    res = await delegate_subagent("do stuff", context=ctx)
    assert "nested" in res.lower()


@pytest.mark.asyncio
async def test_subagent_per_turn_and_concurrency_limits():
    cfg = KesokuConfig()
    cfg.agent.subagent_max_per_turn = 3
    cfg.agent.subagent_max_concurrent = 1
    kctx = KesokuContext(config=cfg)

    ctx = ToolContext(
        session_id="s1",
        is_subagent=False,
        subagents_launched_this_turn=3,
        kesoku_context=kctx,
    )
    res = await delegate_subagent("do stuff", context=ctx)
    assert "max 3 sub-agents per turn reached" in res

    ctx_busy = ToolContext(
        session_id="s1",
        is_subagent=False,
        subagents_launched_this_turn=0,
        running_subagents=1,
        kesoku_context=kctx,
    )
    res_busy = await delegate_subagent("do stuff", context=ctx_busy)
    assert "one sub-agent at a time" in res_busy


@pytest.mark.asyncio
async def test_subagent_executes_tool_round_and_returns_compact_report(tmp_path):
    target_file = tmp_path / "hello.txt"
    target_file.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    cfg = KesokuConfig()
    cfg.agent_working_dir = str(tmp_path)
    cfg.workspace.sessions_dir = str(tmp_path / "sessions")

    llm = SequencedLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCallRequest(
                        name="read_file",
                        arguments={"file_name": str(target_file)},
                        tool_call_id="tc_1",
                    )
                ],
                prompt_tokens=100,
                total_tokens=130,
            ),
            LLMResponse(
                content="Summary: hello.txt has 3 lines (alpha, beta, gamma).",
                tool_calls=[],
                prompt_tokens=150,
                total_tokens=190,
            ),
        ]
    )
    kctx = KesokuContext(config=cfg, llm=llm)
    ctx = ToolContext(
        session_id="sess1",
        session_workspace="sess1",
        kesoku_context=kctx,
    )

    out = await delegate_subagent(
        task="Inspect hello.txt and summarize its lines.",
        context_hint="Look in tmp_path",
        context=ctx,
    )
    assert "2 steps, 1 tool calls, 320 tokens" in out
    assert "Summary: hello.txt has 3 lines" in out
    assert ctx.subagent_llm_calls == 2
    assert ctx.subagent_tokens == 320
    assert ctx.running_subagents == 0
    # Verify only allowed read-only tools were exposed to the sub-agent
    assert {"read_file", "run_shell_command"} <= set(llm.calls[0]["tools"])
    assert "update_file" not in llm.calls[0]["tools"]
    assert "delegate_subagent" not in llm.calls[0]["tools"]


@pytest.mark.asyncio
async def test_subagent_foreground_only_shell_kills_on_timeout(tmp_path):
    cfg = KesokuConfig()
    cfg.agent_working_dir = str(tmp_path)
    cfg.workspace.sessions_dir = str(tmp_path / "sessions")
    cfg.agent.subagent_shell_timeout_seconds = 1
    kctx = KesokuContext(config=cfg)
    ctx = ToolContext(
        session_id="sess1",
        session_workspace="sess1",
        is_subagent=True,
        foreground_only=True,
        kesoku_context=kctx,
    )
    res = await run_shell_command("sleep 3", context=ctx)
    assert "sub-agent commands cannot run in the background" in res
    assert len((await kctx.active_jobs.snapshot("sess1")).running) == 0

