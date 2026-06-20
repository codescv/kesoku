"""Active Context HTML reporter utility for Kesoku AI Agent."""

import html
import json
import logging
import tempfile
from typing import Any

from kesoku.agent.history import prepare_history_for_llm
from kesoku.db import Message, Session, SummaryNode

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _tokenizer = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _tokenizer = None


def estimate_tokens(text: str) -> int:
    """Accurately count tokens using tiktoken (cl100k_base) or fallback to char//4."""
    if not text:
        return 0
    if _tokenizer is not None:
        try:
            return len(_tokenizer.encode(text))
        except Exception:
            pass
    return len(text) // 4


class ContextHtmlReporter:
    """Utility class to render Active Prompt Context into a beautiful interactive HTML page."""

    @staticmethod
    def _render_messages_to_html(msgs: list[Message]) -> str:
        html_bubbles = []
        for msg in msgs:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            mtype = msg.type.value if hasattr(msg.type, "value") else str(msg.type)
            content = msg.content or ""

            # Check role and type mapping
            if role == "user":
                bubble_class = "user"
                role_label = "User"
                safe_content = html.escape(content).replace("\n", "<br>")
            elif role == "system":
                bubble_class = "system"
                role_label = "System"
                safe_content = html.escape(content).replace("\n", "<br>")
            elif role == "assistant":
                if mtype == "thought":
                    bubble_class = "thought"
                    role_label = "Assistant Thought"
                    safe_content = html.escape(content).replace("\n", "<br>")
                else:
                    bubble_class = "assistant"
                    role_label = "Assistant Response"
                    safe_content = html.escape(content).replace("\n", "<br>")
            elif role == "tool":
                if mtype == "tool_call":
                    bubble_class = "tool-call"
                    role_label = "Tool Call"
                    metadata = msg.metadata or {}
                    tool_name = metadata.get("tool_name") or msg.sender or "unknown_tool"
                    tool_args = metadata.get("tool_arguments")
                    if tool_args:
                        try:
                            if isinstance(tool_args, str):
                                args_dict = json.loads(tool_args)
                                arguments_pretty = json.dumps(args_dict, indent=2, ensure_ascii=False)
                            else:
                                arguments_pretty = json.dumps(tool_args, indent=2, ensure_ascii=False)
                        except Exception:
                            arguments_pretty = str(tool_args)
                        safe_content = f"""
                        <div class="tool-call-block" style="margin-top: 0;">
                            🔧 <strong>Called Tool:</strong> <code>{tool_name}</code><br>
                            <pre class="tool-args-pre">{html.escape(arguments_pretty)}</pre>
                        </div>
                        """
                    else:
                        safe_content = html.escape(content).replace("\n", "<br>")
                else:
                    bubble_class = "tool"
                    role_label = "Tool Result"
                    metadata = msg.metadata or {}
                    tool_name = metadata.get("tool_name") or msg.sender or "unknown_tool"
                    safe_content = f"""
                    <div class="tool-response-block" style="margin-top: 0;">
                        📥 <strong>Tool Output (<code>{tool_name}</code>):</strong><br>
                        <pre class="tool-response-pre">{html.escape(content)}</pre>
                    </div>
                    """
            else:
                bubble_class = "system"
                role_label = str(role).capitalize()
                safe_content = html.escape(content).replace("\n", "<br>")

            # Fallback inline tool calls in metadata
            tool_calls = msg.metadata.get("tool_calls") if msg.metadata else None
            if role == "assistant" and tool_calls:
                tool_call_html_parts = []
                for tc in tool_calls:
                    name = tc.get("name") or tc.get("function", {}).get("name", "unknown_tool")
                    arguments = tc.get("arguments") or tc.get("function", {}).get("arguments", "")
                    try:
                        if isinstance(arguments, str):
                            args_dict = json.loads(arguments)
                            arguments_pretty = json.dumps(args_dict, indent=2, ensure_ascii=False)
                        else:
                            arguments_pretty = json.dumps(arguments, indent=2, ensure_ascii=False)
                    except Exception:
                        arguments_pretty = str(arguments)

                    tool_call_html_parts.append(
                        f'<div class="tool-call-block">'
                        f"🔧 <strong>Called Tool:</strong> <code>{name}</code><br>"
                        f'<pre class="tool-args-pre">{html.escape(arguments_pretty)}</pre>'
                        f"</div>"
                    )

                tool_calls_html = "\n".join(tool_call_html_parts)
                if safe_content:
                    safe_content += f"<br><br>{tool_calls_html}"
                else:
                    safe_content = tool_calls_html

            msg_html = f"""
            <div class="chat-bubble {bubble_class}">
                <div class="bubble-content">
                    <strong>{role_label}:</strong><br>
                    {safe_content}
                </div>
            </div>
            """
            html_bubbles.append(msg_html)

        return "\n".join(html_bubbles)

    @staticmethod
    def render_to_temp_file(
        session: Session,
        root_summaries: list[SummaryNode],
        all_summaries: list[SummaryNode],
        protected_head: list[Message],
        buffer: list[Message],
        protected_tail: list[Message],
        sys_msg: str,
        last_metrics: dict[str, Any] | None = None,
    ) -> str:
        """Render the custom context state into a temporary HTML file.

        Args:
            session: Session object context.
            root_summaries: List of active root summary nodes.
            all_summaries: List of all summary nodes (active & parented).
            protected_head: Messages in the protected head.
            buffer: Messages in the middle buffer (pending compaction).
            protected_tail: Messages in the protected tail (fresh tail).
            sys_msg: Resolved system prompt instructions.
            last_metrics: Extracted turn metrics of last assistant execution.

        Returns:
            The absolute path of the generated temporary HTML file.
        """
        # Use shared prepare_history_for_llm logic to clean historical turns
        # Tag each message to trace its original segment boundary after thoughts are stripped.
        tagged_messages = []
        for m in protected_head:
            m_copy = m.model_copy()
            m_copy.metadata = dict(m_copy.metadata) if m_copy.metadata else {}
            m_copy.metadata["_context_group"] = "head"
            tagged_messages.append(m_copy)

        for m in buffer:
            m_copy = m.model_copy()
            m_copy.metadata = dict(m_copy.metadata) if m_copy.metadata else {}
            m_copy.metadata["_context_group"] = "buffer"
            tagged_messages.append(m_copy)

        for m in protected_tail:
            m_copy = m.model_copy()
            m_copy.metadata = dict(m_copy.metadata) if m_copy.metadata else {}
            m_copy.metadata["_context_group"] = "tail"
            tagged_messages.append(m_copy)

        cleaned = prepare_history_for_llm(tagged_messages)

        protected_head = [m for m in cleaned if m.metadata.get("_context_group") == "head"]
        buffer = [m for m in cleaned if m.metadata.get("_context_group") == "buffer"]
        protected_tail = [m for m in cleaned if m.metadata.get("_context_group") == "tail"]

        # Clean up temporary context group tracking metadata
        for segment in (protected_head, buffer, protected_tail):
            for m in segment:
                if m.metadata:
                    m.metadata.pop("_context_group", None)

        # Estimate active context tokens
        total_tokens = (
            estimate_tokens(sys_msg)
            + sum(estimate_tokens(m.content) for m in protected_head)
            + sum(n.token_count for n in root_summaries)
            + sum(estimate_tokens(m.content) for m in buffer)
            + sum(estimate_tokens(m.content) for m in protected_tail)
        )

        # Build summary nodes HTML
        summary_nodes_html = []
        # Sort all summaries by level descending, then start_timestamp ascending
        sorted_summaries = sorted(all_summaries, key=lambda nd: (-nd.level, nd.start_timestamp))
        for node in sorted_summaries:
            is_active = node.parent_id is None
            card_cls = "node-card" if is_active else "node-card inactive"
            badge_cls = f"badge depth-{node.level}" if is_active else "badge inactive"
            active_suffix = "" if is_active else " (Parented / Consolidated)"

            safe_summary = html.escape(node.summary).replace("\n", "<br>")
            savings = round(node.source_token_count / node.token_count, 1) if node.token_count > 0 else 1

            node_html = f"""
            <div class="{card_cls}">
                <div class="node-header">
                    <span class="{badge_cls}">Level-{node.level} Node {node.id[:8]}{active_suffix}</span>
                    <span style="font-size:0.85rem;color:#8899a6;">
                        {node.source_token_count} tokens &rarr; {node.token_count} tokens ({savings}x savings)
                    </span>
                </div>
                <div style="white-space: pre-wrap; font-size:0.95rem;">{safe_summary}</div>
            </div>
            """
            summary_nodes_html.append(node_html)

        summary_nodes_html_str = (
            "\n".join(summary_nodes_html) if summary_nodes_html else "<p>*(No active compacted summary nodes)*</p>"
        )

        # Build message HTML segments
        head_html_str = (
            ContextHtmlReporter._render_messages_to_html(protected_head) or "<p>*(Protected head is empty)*</p>"
        )
        buffer_html_str = (
            ContextHtmlReporter._render_messages_to_html(buffer) or "<p>*(No middle buffer messages)*</p>"
        )
        fresh_tail_html_str = (
            ContextHtmlReporter._render_messages_to_html(protected_tail) or "<p>*(Fresh tail is empty)*</p>"
        )

        # Build actual LLM metrics card if available
        actual_llm_html = ""
        if last_metrics:
            total_llm = last_metrics.get("context_tokens", 0)
            cached_tokens = last_metrics.get("cached_tokens", 0)
            active_tokens = max(0, total_llm - cached_tokens)
            active_k = f"{round(active_tokens / 1000)}K" if active_tokens else "0K"
            cached_k = f"{round(cached_tokens / 1000)}K" if cached_tokens else "0K"
            actual_llm_html = f"""
            <div class="stat-card" style="border-left: 4px solid #10b981;">
                <div class="stat-label">Actual LLM Context (Last Turn)</div>
                <div class="stat-value">{total_llm:,} Tokens</div>
                <div style="font-size: 0.85rem; color: #8899a6; margin-top: 5px;">
                    {active_k} active + {cached_k} cached
                </div>
            </div>
            """

        # Combined HTML / CSS template
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Active Prompt Context - Session {session.id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f1419;
            color: #e1e8ed;
            margin: 0;
            padding: 20px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        header {{
            background-color: #15181c;
            border: 1px solid #2f3336;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        h1 {{
            margin-top: 0;
            font-size: 1.5rem;
            color: #1d9bf0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .stat-card {{
            background-color: #1e2732;
            border-radius: 6px;
            padding: 10px 15px;
            border-left: 4px solid #1d9bf0;
        }}
        .stat-label {{
            font-size: 0.8rem;
            color: #8899a6;
            text-transform: uppercase;
        }}
        .stat-value {{
            font-size: 1.2rem;
            font-weight: bold;
            margin-top: 5px;
        }}
        details {{
            background-color: #15181c;
            border: 1px solid #2f3336;
            border-radius: 8px;
            margin-bottom: 15px;
            padding: 15px;
        }}
        summary {{
            font-weight: bold;
            cursor: pointer;
            outline: none;
            user-select: none;
            color: #1d9bf0;
        }}
        pre {{
            background-color: #000000;
            color: #00ff00;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
            font-family: "Fira Code", Consolas, Monaco, monospace;
            font-size: 0.9rem;
            border: 1px solid #202327;
            white-space: pre-wrap;
        }}
        .node-card {{
            background-color: #1e2732;
            border: 1px solid #2f3336;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 10px;
        }}
        .node-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #2f3336;
            padding-bottom: 8px;
            margin-bottom: 10px;
        }}
        .badge {{
            background-color: #1d9bf0;
            color: #ffffff;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        .badge.depth-0 {{ background-color: #10b981; }}
        .badge.depth-1 {{ background-color: #f59e0b; }}
        .badge.depth-2 {{ background-color: #ef4444; }}
        .node-card.inactive {{
            opacity: 0.55;
            border-style: dashed;
            background-color: #161c23;
        }}
        .badge.inactive {{
            background-color: #536471 !important;
            color: #f7f9f9;
        }}
        .chat-bubble {{
            display: flex;
            flex-direction: column;
            margin-bottom: 15px;
            max-width: 85%;
        }}
        .chat-bubble.user {{
            margin-left: auto;
            align-items: flex-end;
        }}
        .chat-bubble.assistant {{
            margin-right: auto;
            align-items: flex-start;
        }}
        .chat-bubble.thought {{
            margin-right: auto;
            align-items: flex-start;
        }}
        .chat-bubble.tool {{
            margin-right: auto;
            align-items: flex-start;
            width: 100%;
        }}
        .chat-bubble.tool-call {{
            margin-right: auto;
            align-items: flex-start;
            width: 100%;
        }}
        .bubble-content {{
            padding: 12px 16px;
            border-radius: 18px;
            font-size: 0.95rem;
        }}
        .chat-bubble.user .bubble-content {{
            background-color: #1d9bf0;
            color: #ffffff;
            border-bottom-right-radius: 4px;
        }}
        .chat-bubble.assistant .bubble-content {{
            background-color: #2f3336;
            color: #e1e8ed;
            border-bottom-left-radius: 4px;
        }}
        .chat-bubble.thought .bubble-content {{
            background-color: #1e1e1e;
            color: #8899a6;
            border: 1px dashed #444;
            border-bottom-left-radius: 4px;
        }}
        .chat-bubble.tool .bubble-content {{
            background-color: #14221f;
            color: #e1e8ed;
            border-bottom-left-radius: 4px;
            width: 100%;
            box-sizing: border-box;
        }}
        .chat-bubble.tool-call .bubble-content {{
            background-color: #1b2836;
            color: #e1e8ed;
            border-left: 4px solid #f59e0b;
            border-bottom-left-radius: 4px;
            box-sizing: border-box;
            width: 100%;
        }}
        .tool-call-block {{
            background-color: #1e2732;
            border: 1px solid #2f3336;
            border-radius: 8px;
            padding: 12px;
            margin-top: 10px;
            font-size: 0.9rem;
            align-self: stretch;
        }}
        .tool-response-block {{
            background-color: #14221f;
            border: 1px solid #10b981;
            border-radius: 8px;
            padding: 12px;
            margin-top: 10px;
            font-size: 0.9rem;
            align-self: stretch;
        }}
        .tool-response-pre {{
            background-color: #0d1512;
            color: #34d399;
            padding: 8px;
            border-radius: 4px;
            font-family: "Fira Code", Consolas, Monaco, monospace;
            font-size: 0.85rem;
            margin: 8px 0 0 0;
            border: 1px solid #10b981;
            white-space: pre-wrap;
        }}
        .tool-args-pre {{
            background-color: #0f1419;
            color: #ffaa00;
            padding: 8px;
            border-radius: 4px;
            font-family: "Fira Code", Consolas, Monaco, monospace;
            font-size: 0.85rem;
            margin: 8px 0 0 0;
            border: 1px solid #2f3336;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📖 Active Prompt Context</h1>
            <p style="margin:0;color:#8899a6;">Session: <strong>{session.id}</strong></p>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Summaries</div>
                    <div class="stat-value">{len(all_summaries)} Nodes ({len(root_summaries)} Active)</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Middle Buffer</div>
                    <div class="stat-value">{len(buffer)} Messages</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Protected Front/Tail</div>
                    <div class="stat-value">{len(protected_head)} / {len(protected_tail)} Messages</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Chat History Size (Est.)</div>
                    <div class="stat-value">{total_tokens:,} Tokens</div>
                </div>
                {actual_llm_html}
            </div>
        </header>

        <details>
            <summary>🛠️ System Message Instructions</summary>
            <pre>{html.escape(sys_msg)}</pre>
        </details>

        <details>
            <summary>🛡️ Protected Front Head (First Turn)</summary>
            <div style="margin-top: 15px; display: flex; flex-direction: column;">
                {head_html_str}
            </div>
        </details>

        <details open>
            <summary>📦 Compacted Summary Scaffold (Hierarchy Forest)</summary>
            <div style="margin-top: 15px;">
                {summary_nodes_html_str}
            </div>
        </details>

        <details open>
            <summary>⏳ Active Buffer (Pending Compaction)</summary>
            <div style="margin-top: 15px; display: flex; flex-direction: column;">
                {buffer_html_str}
            </div>
        </details>

        <details open>
            <summary>🧵 Active Fresh Tail (Chronological Messages)</summary>
            <div style="margin-top: 15px; display: flex; flex-direction: column;">
                {fresh_tail_html_str}
            </div>
        </details>
    </div>
</body>
</html>"""

        try:
            temp_file = tempfile.NamedTemporaryFile(
                suffix="_active_context.html", delete=False, mode="w", encoding="utf-8"
            )
            temp_file.write(html_template)
            temp_file.close()
            return temp_file.name
        except Exception as e:
            logger.error(f"Failed to write Active Context HTML: {e}")
            raise
