"""Unit tests for History cleaning, optimization, and OpenLCM mapping translations."""

from kesoku.agent.history import messages_to_openlcm_dicts, openlcm_dicts_to_messages
from kesoku.constants import MessageRole, MessageType
from kesoku.db import Message


def test_messages_to_openlcm_dicts_simple() -> None:
    """Verify messages_to_openlcm_dicts correctly translates basic messages."""
    history = [
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="User",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Hello, Kesoku!",
        ),
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="Hello, human!",
        ),
    ]

    dicts = messages_to_openlcm_dicts(history)
    assert len(dicts) == 2
    assert dicts[0] == {"role": "user", "content": "Hello, Kesoku!"}
    assert dicts[1] == {"role": "assistant", "content": "Hello, human!"}


def test_messages_to_openlcm_dicts_tool_consolidation() -> None:
    """Verify that consecutive assistant thoughts and tool calls are consolidated."""
    history = [
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.THOUGHT,
            content="Thinking about adding numbers.",
        ),
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="Kesoku",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_CALL,
            content="Calling add with args x=5, y=5",
            metadata={
                "tool_name": "add_numbers",
                "tool_arguments": {"x": 5, "y": 5},
                "tool_call_id": "tc_1",
            },
        ),
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="add_numbers",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_RESULT,
            content="10",
            metadata={
                "tool_name": "add_numbers",
                "tool_result": "10",
                "tool_call_id": "tc_1",
            },
        ),
    ]

    dicts = messages_to_openlcm_dicts(history)
    assert len(dicts) == 2

    # The first assistant thought + tool call are merged
    assert dicts[0]["role"] == "assistant"
    assert dicts[0]["content"] == "Thinking about adding numbers."
    assert len(dicts[0]["tool_calls"]) == 1
    assert dicts[0]["tool_calls"][0] == {
        "id": "tc_1",
        "type": "function",
        "function": {
            "name": "add_numbers",
            "arguments": '{"x": 5, "y": 5}',
        },
    }

    # The tool result remains a separate tool role message
    assert dicts[1] == {
        "role": "tool",
        "content": "10",
        "tool_call_id": "tc_1",
        "name": "add_numbers",
    }


def test_openlcm_dicts_to_messages_unpacking() -> None:
    """Verify that consolidated assistant messages are correctly unpacked back."""
    dicts = [
        {
            "role": "assistant",
            "content": "Thinking about adding numbers.",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "arguments": '{"x": 5, "y": 5}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": "10",
            "tool_call_id": "tc_1",
            "name": "add_numbers",
        },
        {
            "role": "assistant",
            "content": "The result is 10.",
        },
    ]

    msgs = openlcm_dicts_to_messages(
        dicts,
        session_id="sess1",
        chatbot_id="cli",
        channel_id="chan1",
    )

    # Should unpack first block into 1 thought and 1 tool call message
    assert len(msgs) == 4

    # Msg 0: Thought
    assert msgs[0].role == MessageRole.ASSISTANT
    assert msgs[0].type == MessageType.THOUGHT
    assert msgs[0].content == "Thinking about adding numbers."

    # Msg 1: Tool call
    assert msgs[1].role == MessageRole.TOOL
    assert msgs[1].type == MessageType.TOOL_CALL
    assert msgs[1].metadata["tool_name"] == "add_numbers"
    assert msgs[1].metadata["tool_arguments"] == {"x": 5, "y": 5}
    assert msgs[1].metadata["tool_call_id"] == "tc_1"

    # Msg 2: Tool result
    assert msgs[2].role == MessageRole.TOOL
    assert msgs[2].type == MessageType.TOOL_RESULT
    assert msgs[2].content == "10"
    assert msgs[2].metadata["tool_call_id"] == "tc_1"

    # Msg 3: Assistant response
    assert msgs[3].role == MessageRole.ASSISTANT
    assert msgs[3].type == MessageType.TEXT
    assert msgs[3].content == "The result is 10."


def test_messages_to_openlcm_dicts_thought_text_merging() -> None:
    """Verify thought and text are merged into a single assistant message using tags."""
    history = [
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.THOUGHT,
            content="My thought process here.",
        ),
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="Hello, user!",
        ),
    ]

    dicts = messages_to_openlcm_dicts(history)
    assert len(dicts) == 1
    assert dicts[0] == {
        "role": "assistant",
        "content": "<thought>My thought process here.</thought>\n\nHello, user!",
    }


def test_openlcm_dicts_to_messages_thought_text_unpacking() -> None:
    """Verify that merged assistant messages with thought tags are parsed and unpacked."""
    dicts = [
        {
            "role": "assistant",
            "content": "<thought>My thought process here.</thought>\n\nHello, user!",
        }
    ]

    msgs = openlcm_dicts_to_messages(
        dicts,
        session_id="sess1",
        chatbot_id="cli",
        channel_id="chan1",
    )

    assert len(msgs) == 2

    assert msgs[0].role == MessageRole.ASSISTANT
    assert msgs[0].type == MessageType.THOUGHT
    assert msgs[0].content == "My thought process here."

    assert msgs[1].role == MessageRole.ASSISTANT
    assert msgs[1].type == MessageType.TEXT
    assert msgs[1].content == "Hello, user!"


def test_thought_signature_preservation() -> None:
    """Verify that thought_signature is preserved through OpenLCM conversions."""
    history = [
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="Kesoku",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_CALL,
            content="Calling tool",
            metadata={
                "tool_name": "test_tool",
                "tool_arguments": {"arg": 1},
                "tool_call_id": "tc_1",
                "thought_signature": "sig_123456",
            },
        )
    ]

    # 1. Convert to OpenLCM dicts
    dicts = messages_to_openlcm_dicts(history)
    assert len(dicts) == 1
    assert dicts[0]["role"] == "assistant"
    assert len(dicts[0]["tool_calls"]) == 1
    assert dicts[0]["tool_calls"][0]["thought_signature"] == "sig_123456"

    # 2. Convert back to Messages
    msgs = openlcm_dicts_to_messages(
        dicts,
        session_id="sess1",
        chatbot_id="cli",
        channel_id="chan1",
    )
    assert len(msgs) == 1
    assert msgs[0].role == MessageRole.TOOL
    assert msgs[0].type == MessageType.TOOL_CALL
    assert msgs[0].metadata.get("thought_signature") == "sig_123456"


