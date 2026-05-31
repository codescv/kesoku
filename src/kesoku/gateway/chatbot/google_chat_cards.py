"""Card builders for Google Chat integration."""

import html
from typing import Any


class GoogleChatCardBuilder:
    """Constructs cardsV2 structures for Google Chat messages."""

    @staticmethod
    def get_tool_arguments_suffix(tool_arguments: dict[str, Any] | None) -> str:
        """Format and retrieve the tool arguments suffix for Google Chat card display.

        Args:
            tool_arguments: The tool arguments dictionary.

        Returns:
            Formatted suffix string (e.g., ': <code>arg_value</code>'), or empty string if none.
        """
        if not tool_arguments:
            return ""

        arg_str = ""
        if isinstance(tool_arguments, dict):
            # Exclude framework/context arguments
            filtered_args = {k: v for k, v in tool_arguments.items() if k != "context"}
            if len(filtered_args) == 1:
                val = next(iter(filtered_args.values()))
                arg_str = str(val)
            elif len(filtered_args) > 1:
                arg_str = ", ".join(f"{k}: {v}" for k, v in filtered_args.items())

        if arg_str:
            arg_str = arg_str.replace("\n", " ")
            if len(arg_str) > 80:
                arg_str = arg_str[:80] + "..."

        return f": <code>{html.escape(arg_str)}</code>" if arg_str else ""

    @staticmethod
    def build_foldable_ui_card(
        session_id: str,
        items: list[dict[str, Any]],
        status: str = "running",
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Construct a single foldable UI card for all intermediate thoughts and tools.

        Args:
            session_id: Active session ID.
            items: List of intermediate special messages/thoughts/tools.
            status: Either 'running', 'finished', or 'interrupted'.
            metrics: Optional dictionary containing session and turn metrics.

        Returns:
            A cardsV2 dictionary structure.
        """
        widgets = []
        for item in items:
            if item["type"] == "thought":
                content_html = html.escape(item["content"]).replace("\n", "<br>")
                widgets.append({"textParagraph": {"text": f"💭 <b>Thought:</b> {content_html}"}})
            elif item["type"] == "tool_call":
                emoji = item["status"]
                widgets.append({"textParagraph": {"text": f"🛠️ <b>{item['tool_name']}</b>{item['arg_suffix']} {emoji}"}})
            elif item["type"] == "system":
                content_html = html.escape(item["content"]).replace("\n", "<br>")
                widgets.append({"textParagraph": {"text": f"⚙️ <b>System:</b> {content_html}"}})

        if not widgets:
            widgets.append({"textParagraph": {"text": "<i>Preparing turn...</i>"}})

        # Collapsible section for Thoughts & Tools
        thoughts_tools_section = {
            "header": "Thoughts & Tools",
            "collapsible": True,
            "uncollapsibleWidgetsCount": 0,
            "widgets": widgets,
        }

        card_sections = [thoughts_tools_section]

        # Control / Metrics section
        if status in ("finished", "interrupted") and metrics:
            session_turns = metrics.get("session_turns", 0)
            context_tokens = metrics.get("context_tokens", 0)
            turn_tool_calls = metrics.get("turn_tool_calls", 0)
            turn_tokens = metrics.get("turn_tokens", 0)
            turn_time = metrics.get("turn_time", 0.0)

            context_k = f"{round(context_tokens / 1000)}K"
            turn_k = f"{round(turn_tokens / 1000)}K"

            if status == "finished":
                prefix = "⚡"
                suffix = ""
            else:
                prefix = "🛑"
                suffix = " (Interrupted)"

            metrics_text = (
                f"{prefix} <b>Session:</b> {session_turns} turns | <b>Context:</b> {context_k} tokens{suffix}<br>"
                f"⏱️ <b>Turn:</b> {turn_tool_calls} tool calls | {turn_k} tokens | {turn_time:.1f}s"
            )
            card_sections.append({"widgets": [{"textParagraph": {"text": metrics_text}}]})

        return {
            "cardId": f"foldable_ui_{session_id}",
            "card": {
                "header": {
                    "title": "Kesoku Agent",
                    "subtitle": "Active Turn" if status == "running" else "Turn Completed",
                },
                "sections": card_sections,
            },
        }

    @staticmethod
    def build_question_card(session_id: str, question: str, choices: list[str]) -> dict[str, Any]:
        """Construct the multiple-choice question card without interactive buttons.

        Args:
            session_id: Active session ID.
            question: The question prompt text.
            choices: A list of choice values.

        Returns:
            A cardsV2 dictionary structure.
        """
        choices_list = "\n".join(f"- {choice}" for choice in choices)
        card_text = f"{question}\n\n{choices_list}"
        return {
            "cardId": f"question_{session_id}",
            "card": {
                "sections": [
                    {
                        "widgets": [
                            {
                                "textParagraph": {
                                    "text": card_text,
                                    "textSyntax": "MARKDOWN",
                                }
                            }
                        ]
                    },
                ],
            },
        }
