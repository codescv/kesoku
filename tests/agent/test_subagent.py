from unittest.mock import MagicMock

import pytest

from kesoku.agent.tools.registry import ToolContext
from kesoku.agent.tools.subagent import delegate_subagent


class MockLLM:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
        self.model_name = "mock"
    async def generate_response(self, *args, **kwargs):
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp

@pytest.mark.asyncio
async def test_subagent_tool_rejection():
    # Test that delegate_subagent rejects if is_subagent is True
    ctx = MagicMock(spec=ToolContext)
    ctx.is_subagent = True

    res = await delegate_subagent("do stuff", context=ctx)
    assert "nested" in res.lower()

@pytest.mark.asyncio
async def test_subagent_concurrency_limit():
    ctx = MagicMock(spec=ToolContext)
    ctx.is_subagent = False
    ctx.subagents_launched_this_turn = 0

    # max_per_turn = 3, running 4th should fail
    from kesoku.config import get_config
    cfg = get_config()
    cfg.agent.subagent_max_per_turn = 3

    ctx.subagents_launched_this_turn = 3
    res = await delegate_subagent("do stuff", context=ctx)
    assert "max" in res.lower()
