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


def test_messages_to_openlcm_dicts_strips_historical_thoughts() -> None:
    """Verify that messages_to_openlcm_dicts strips thoughts from completed turns only."""
    history = [
        # Turn 1 (Completed)
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="User",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Hello",
        ),
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.THOUGHT,
            content="Historical thought",
        ),
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="Hi there",
        ),
        # Turn 2 (Active/Latest)
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="User",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="How are you?",
        ),
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.THOUGHT,
            content="Active thought",
        ),
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="I am good",
        ),
    ]

    dicts = messages_to_openlcm_dicts(history)
    # We should have:
    # 1. user: "Hello"
    # 2. assistant: "Hi there" (without "Historical thought")
    # 3. user: "How are you?"
    # 4. assistant: "<thought>Active thought</thought>\n\nI am good"
    assert len(dicts) == 4
    assert dicts[0]["role"] == "user"
    assert dicts[0]["content"] == "Hello"
    assert dicts[1]["role"] == "assistant"
    assert dicts[1]["content"] == "Hi there"
    assert dicts[2]["role"] == "user"
    assert dicts[2]["content"] == "How are you?"
    assert dicts[3]["role"] == "assistant"
    assert dicts[3]["content"] == "<thought>Active thought</thought>\n\nI am good"


def test_path_sanitization_and_restoration() -> None:
    """Verify that absolute paths under sessions staging dir are sanitized and restored."""
    history = [
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT,
            content="Check file: /path/to/sessions/sess1/file.png",
        ),
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
                "tool_arguments": {"path": "/path/to/sessions/sess1/file.png"},
                "tool_call_id": "tc_1",
            },
        ),
    ]

    # 1. Sanitize (to OpenLCM dicts)
    dicts = messages_to_openlcm_dicts(history)
    assert len(dicts) == 2
    assert dicts[0]["content"] == "Check file: $STAGING_DIR/file.png"
    assert "tool_calls" in dicts[1]
    import json
    args = json.loads(dicts[1]["tool_calls"][0]["function"]["arguments"])
    assert args["path"] == "$STAGING_DIR/file.png"

    # 2. Restore (back to Messages)
    msgs = openlcm_dicts_to_messages(
        dicts,
        session_id="sess1",
        chatbot_id="cli",
        channel_id="chan1",
        workspace_name="sess1",
    )
    assert len(msgs) == 2
    # Verify replaced path contains the correct staging dir path
    assert "sessions/sess1/file.png" in msgs[0].content
    assert msgs[1].role == MessageRole.TOOL
    assert msgs[1].type == MessageType.TOOL_CALL
    assert "sessions/sess1/file.png" in msgs[1].metadata["tool_arguments"]["path"]


def test_messages_to_openlcm_dicts_tool_sorting() -> None:
    """Verify that tool results are sorted to match the order of tool calls."""
    history = [
        Message(
            id="call_1",
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="Kesoku",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_CALL,
            content="Call 1",
            metadata={"tool_name": "tool_1", "tool_call_id": "tc_1"},
        ),
        Message(
            id="call_2",
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="Kesoku",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_CALL,
            content="Call 2",
            metadata={"tool_name": "tool_2", "tool_call_id": "tc_2"},
        ),
        # Out of order results: result 2 finishes and is recorded first
        Message(
            id="res_2",
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="tool_2",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_RESULT,
            content="Result 2",
            parent_id="call_2",
            metadata={"tool_name": "tool_2", "tool_result": "r2", "tool_call_id": "tc_2"},
        ),
        Message(
            id="res_1",
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="tool_1",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_RESULT,
            content="Result 1",
            parent_id="call_1",
            metadata={"tool_name": "tool_1", "tool_result": "r1", "tool_call_id": "tc_1"},
        ),
    ]

    dicts = messages_to_openlcm_dicts(history)

    assert len(dicts) == 3
    assert dicts[0]["role"] == "assistant"
    assert len(dicts[0]["tool_calls"]) == 2
    assert dicts[0]["tool_calls"][0]["id"] == "tc_1"
    assert dicts[0]["tool_calls"][1]["id"] == "tc_2"

    assert dicts[1]["role"] == "tool"
    assert dicts[1]["tool_call_id"] == "tc_1"
    assert dicts[1]["content"] == "r1"

    assert dicts[2]["role"] == "tool"
    assert dicts[2]["tool_call_id"] == "tc_2"
    assert dicts[2]["content"] == "r2"


def test_messages_to_openlcm_dicts_active_in_progress_turn_preserves_thought() -> None:
    """Verify that thoughts are preserved for the latest turn if it is in-progress (no TEXT response)."""
    history = [
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="User",
            role=MessageRole.USER,
            type=MessageType.TEXT,
            content="Hello",
        ),
        # Active Turn (In-progress)
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="Kesoku",
            role=MessageRole.ASSISTANT,
            type=MessageType.THOUGHT,
            content="Running a tool now",
        ),
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="call_1",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_CALL,
            content="Calling tool...",
            id="call_1",
            metadata={"tool_name": "tool_1", "tool_arguments": {}},
        ),
        Message(
            session_id="sess1",
            chatbot_id="cli",
            channel_id="chan1",
            sender="tool_1",
            role=MessageRole.TOOL,
            type=MessageType.TOOL_RESULT,
            content="Result 1",
            parent_id="call_1",
            metadata={"tool_name": "tool_1", "tool_result": "r1", "tool_call_id": "tc_1"},
        ),
    ]

    dicts = messages_to_openlcm_dicts(history)

    # We should have:
    # 1. user: "Hello"
    # 2. assistant: "Running a tool now" with tool_calls
    # 3. tool: "r1"
    assert len(dicts) == 3
    assert dicts[0]["role"] == "user"
    assert dicts[1]["role"] == "assistant"
    # The thought content must be preserved as the content of the assistant message containing the tool calls
    assert dicts[1]["content"] == "Running a tool now"
    assert len(dicts[1]["tool_calls"]) == 1
    assert dicts[2]["role"] == "tool"
    assert dicts[2]["content"] == "r1"




