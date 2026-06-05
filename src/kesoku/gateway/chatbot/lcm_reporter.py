"""LCM Active Context HTML reporter utility for Kesoku AI Agent."""

import html
import json
import logging
import tempfile
from collections import defaultdict
from typing import Any

from openlcm.core.tokens import count_messages_tokens

from kesoku.db import Session

logger = logging.getLogger(__name__)


class LcmHtmlReporter:
    """Utility class to render LCM Active Context into a beautiful interactive HTML page."""

    @staticmethod
    def render_to_temp_file(
        session: Session,
        all_nodes: list[Any],
        active_node_ids: set[int],
        fresh_msgs: list[dict[str, Any]],
        sys_msg: str,
        assembled_context: list[dict[str, Any]],
    ) -> str:
        """Render the LCM context state into a temporary HTML file.

        Args:
            session: Session object context.
            all_nodes: All DAG nodes in the session.
            active_node_ids: Set of active node IDs.
            fresh_msgs: Message dicts in the fresh tail.
            sys_msg: Resolved system prompt instructions.
            assembled_context: Combined/assembled context message list.

        Returns:
            The absolute path of the generated temporary HTML file.
        """
        total_tokens = count_messages_tokens(assembled_context)

        # Build summary nodes HTML
        summary_nodes_html = []
        by_depth = defaultdict(list)
        for node in all_nodes:
            by_depth[node.depth].append(node)

        max_dag_depth = max(by_depth.keys()) if by_depth else -1
        for depth in range(max_dag_depth, -1, -1):
            nodes_at_depth = sorted(by_depth.get(depth, []), key=lambda nd: nd.created_at)
            depth_label = {0: "Recent", 1: "Session Arc", 2: "Durable"}.get(depth, f"Depth-{depth}")
            for node in nodes_at_depth:
                is_active = node.node_id in active_node_ids
                card_cls = "node-card" if is_active else "node-card inactive"
                badge_cls = f"badge depth-{node.depth}" if is_active else "badge inactive"
                active_suffix = "" if is_active else " (Condensed)"

                safe_summary = html.escape(node.summary).replace("\n", "<br>")
                savings = round(node.source_token_count / node.token_count, 1) if node.token_count > 0 else 1
                hint_text = node.expand_hint or f"lcm_expand(node_id={node.node_id}) to retrieve details"
                safe_hint = html.escape(hint_text)

                node_html = f"""
                <div class="{card_cls}">
                    <div class="node-header">
                        <span class="{badge_cls}">{depth_label} Node {node.node_id}{active_suffix}</span>
                        <span style="font-size:0.85rem;color:#8899a6;">
                            {node.source_token_count} tokens &rarr; {node.token_count} tokens ({savings}x savings)
                        </span>
                    </div>
                    <div style="white-space: pre-wrap; font-size:0.95rem;">{safe_summary}</div>
                    <div style="font-size:0.8rem;color:#1d9bf0;margin-top:8px;font-style:italic;">
                        Hint: {safe_hint}
                    </div>
                </div>
                """
                summary_nodes_html.append(node_html)

        summary_nodes_html_str = (
            "\n".join(summary_nodes_html) if summary_nodes_html else "<p>*(No active compacted summary nodes)*</p>"
        )

        # Build fresh tail HTML
        fresh_tail_html = []
        for msg in fresh_msgs:
            role = msg.get("role")
            content = msg.get("content") or ""

            bubble_class = (
                "user"
                if role == "user"
                else "assistant"
                if role == "assistant"
                else "tool"
                if role == "tool"
                else "system"
            )
            if role == "user":
                role_label = "User"
            elif role == "assistant":
                role_label = "Assistant"
            elif role == "tool":
                role_label = "Tool Result"
            else:
                role_label = str(role).capitalize()

            if role == "tool":
                tool_name = msg.get("name") or "unknown_tool"
                safe_content = f"""
                <div class="tool-response-block">
                    📥 <strong>Tool Output (<code>{tool_name}</code>):</strong><br>
                    <pre class="tool-response-pre">{html.escape(content)}</pre>
                </div>
                """
            else:
                safe_content = html.escape(content).replace("\n", "<br>")

            # Display embedded tool calls if present inside the assistant role message
            if role == "assistant" and msg.get("tool_calls"):
                tool_call_html_parts = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown_tool")
                    arguments = fn.get("arguments", "")
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
            fresh_tail_html.append(msg_html)

        fresh_tail_html_str = "\n".join(fresh_tail_html) if fresh_tail_html else "<p>*(Fresh tail is empty)*</p>"

        # Combined HTML / CSS template
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LCM Active Context - Session {session.id}</title>
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
        .chat-bubble.tool {{
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
        .chat-bubble.tool .bubble-content {{
            background-color: #14221f;
            color: #e1e8ed;
            border-bottom-left-radius: 4px;
            width: 100%;
            box-sizing: border-box;
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
            <h1>📖 Lossless Context Management (LCM) Active Context</h1>
            <p style="margin:0;color:#8899a6;">Session: <strong>{session.id}</strong></p>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Summaries</div>
                    <div class="stat-value">{len(all_nodes)} Nodes</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Fresh Tail</div>
                    <div class="stat-value">{len(fresh_msgs)} Messages</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Active Context Size</div>
                    <div class="stat-value">{total_tokens} Tokens</div>
                </div>
            </div>
        </header>

        <details>
            <summary>🛠️ System Message Instructions</summary>
            <pre>{html.escape(sys_msg)}</pre>
        </details>

        <details open>
            <summary>📦 Compacted Summary Scaffold (DAG Nodes)</summary>
            <div style="margin-top: 15px;">
                {summary_nodes_html_str}
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
                suffix="_lcm_context.html", delete=False, mode="w", encoding="utf-8"
            )
            temp_file.write(html_template)
            temp_file.close()
            return temp_file.name
        except Exception as e:
            logger.error(f"Failed to write LCM context HTML: {e}")
            raise
