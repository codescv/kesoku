"""Discord UI components for Kesoku AI Agent chatbot.

Provides interactive UI elements such as persistent views and html trajectory viewers.
"""

import asyncio
import datetime
import html
import io
from typing import Any

import tzlocal

import discord
from kesoku.agent.history import build_history
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger
from kesoku.utils.async_fs import async_exists

logger = setup_logger(__name__)


class MessageHeaderView(discord.ui.View):
    """Persistent Discord view representing the conversation header with interactive trajectory viewer."""

    def __init__(self, gateway: Gateway, session_id: str, chatbot: Any = None, is_thread: bool = False) -> None:
        """Initialize the MessageHeaderView.

        Args:
            gateway: The Kesoku Gateway instance.
            session_id: Session ID of the conversation.
            chatbot: Optional reference to the active Discord chatbot.
            is_thread: True if the current conversation channel is a thread.
        """
        super().__init__(timeout=None)
        self.gateway = gateway
        self.session_id = session_id
        self.chatbot = chatbot
        self.is_thread = is_thread

        # Clear session button should not be visible inside thread sessions
        if is_thread:
            self.remove_item(self.clear_session)

    @discord.ui.button(
        style=discord.ButtonStyle.secondary,
        emoji="📜",
        custom_id="btn_view_trajectory",
    )
    async def view_trajectory(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        """Callback triggered when 'View Trajectory' button is clicked."""
        await interaction.response.defer(ephemeral=True)

        try:
            # Fetch entire clean historical context in grouped user-facing order
            history = await build_history(
                gateway=self.gateway,
                session_id=self.session_id,
                order="grouped",
                heal_orphans=False,
            )

            # Fetch system prompt directly from session and prepend as first item
            session = await self.gateway.db.get_session(self.session_id)
            if session and session.system_prompt:
                sys_msg = Message(
                    session_id=self.session_id,
                    chatbot_id="system",
                    channel_id="system",
                    sender="System",
                    role=MessageRole.SYSTEM,
                    type=MessageType.TEXT,
                    content=session.system_prompt,
                    status=MessageStatus.RESPONDED,
                    timestamp=session.created_at - 0.01,
                )
                history.insert(0, sys_msg)

            # Generate beautiful HTML trajectory content
            html_content = self._generate_html_trajectory(history)

            # Stream in-memory buffer to Discord as a file attachment
            file_data = io.BytesIO(html_content.encode("utf-8"))
            discord_file = discord.File(fp=file_data, filename="trajectory.html")

            await interaction.followup.send(
                content="Here is the complete interactive trace of the conversation turn:",
                file=discord_file,
                ephemeral=True,
            )
        except Exception as e:
            logger.error(f"Failed to generate trajectory for session {self.session_id}: {e}", exc_info=True)
            await interaction.followup.send(
                content=f"⚠️ Failed to generate trajectory: {e}",
                ephemeral=True,
            )

    @discord.ui.button(
        style=discord.ButtonStyle.secondary,
        emoji="📖",
        custom_id="btn_view_lcm_context",
    )
    async def view_lcm_context(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        """Callback triggered when 'View LCM Context' button is clicked."""
        await interaction.response.defer(ephemeral=True)

        try:
            if not self.chatbot:
                await interaction.followup.send(
                    content="⚠️ Discord chatbot adapter reference is unavailable.",
                    ephemeral=True,
                )
                return

            res = await self.chatbot.get_session_lcm_context_by_channel(str(interaction.channel_id))
            if await async_exists(res):
                discord_file = discord.File(res, filename="lcm_active_context.html")
                await interaction.followup.send(
                    content="📖 Here is the complete lossless active context (what the Agent currently sees):",
                    file=discord_file,
                    ephemeral=True,
                )
            else:
                await interaction.followup.send(
                    content=res,
                    ephemeral=True,
                )
        except Exception as e:
            logger.error(f"Failed to retrieve LCM context via button for session {self.session_id}: {e}", exc_info=True)
            await interaction.followup.send(
                content=f"⚠️ Failed to retrieve LCM context: {e}",
                ephemeral=True,
            )

    @discord.ui.button(
        style=discord.ButtonStyle.secondary,
        emoji="🛑",
        custom_id="btn_stop_turn",
    )
    async def stop_turn(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        """Callback triggered when 'Stop' button is clicked.

        Aborts the current active agent turn, cancels the typing spinner, and deletes intermediate messages.
        """
        await interaction.response.defer(ephemeral=True)

        try:
            # 1. Locate the active dispatcher agent and stop the session worker immediately
            agent = self.gateway.agent
            if agent:
                await agent.stop_session_worker(self.session_id, immediate=True)

            # Allow the cancelled worker to write the turn metrics to database
            await asyncio.sleep(0.15)

            # 2. Fetch recent history to find the active user message and update its status
            history = await self.gateway.db.get_session_history(self.session_id, limit=20)
            user_msg = None
            for msg in reversed(history):
                if msg.role == MessageRole.USER:
                    user_msg = msg
                    break

            if user_msg:
                if user_msg.status in (MessageStatus.PENDING_AGENT, MessageStatus.PROCESSING):
                    await self.gateway.db.update_message_status(user_msg.id, MessageStatus.INTERRUPTED)

            # Delete the message header entirely on interruption
            try:
                await interaction.message.delete()
            except Exception as de:
                logger.warning(f"Failed to delete header message on stop: {de}")

            # 3. Stop typing task and clean up intermediate special messages in Discord UI
            if self.chatbot:
                channel_id_str = str(interaction.channel_id)
                typing_task = self.chatbot._typing_tasks.pop(channel_id_str, None)
                if typing_task:
                    typing_task.cancel()

                intermediate_msgs = self.chatbot._intermediate_messages.pop(channel_id_str, [])
                if intermediate_msgs:
                    for msg in intermediate_msgs:
                        try:
                            await msg.delete()
                        except Exception as de:
                            logger.warning(f"Failed to delete intermediate message {msg.id}: {de}")

                # Clear session reference from chatbot's header views cache
                for tid, (header_msg, _) in list(self.chatbot._header_views.items()):
                    if header_msg.id == interaction.message.id:
                        self.chatbot._header_views.pop(tid, None)
                        break

                self.chatbot._turn_special_items.pop(self.session_id, None)
                self.chatbot._turn_special_msg.pop(self.session_id, None)

            await interaction.followup.send(
                content="🛑 The agent turn was stopped, and intermediate special messages were removed.",
                ephemeral=True,
            )
        except Exception as e:
            logger.error(f"Failed to stop turn for session {self.session_id}: {e}", exc_info=True)
            await interaction.followup.send(
                content=f"⚠️ Failed to stop turn: {e}",
                ephemeral=True,
            )

    @discord.ui.button(
        style=discord.ButtonStyle.secondary,
        emoji="♻️",
        custom_id="btn_clear_session",
    )
    async def clear_session(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        """Callback triggered when 'Clear Session' button is clicked.

        Aborts any active turn, deletes database records and workspace from disk, and cancels UI elements.
        """
        await interaction.response.defer(ephemeral=True)

        try:
            # 1. Locate the active dispatcher agent and stop the session worker immediately if running
            agent = self.gateway.agent
            if agent:
                await agent.stop_session_worker(self.session_id, immediate=True)

            # 2. Delete the session and its history/workspace via Gateway
            await self.gateway.delete_session(self.session_id)

            # 3. Stop typing task and clean up intermediate special messages in Discord UI
            if self.chatbot:
                channel_id_str = str(interaction.channel_id)
                typing_task = self.chatbot._typing_tasks.pop(channel_id_str, None)
                if typing_task:
                    typing_task.cancel()

                intermediate_msgs = self.chatbot._intermediate_messages.pop(channel_id_str, [])
                if intermediate_msgs:
                    for msg in intermediate_msgs:
                        try:
                            await msg.delete()
                        except Exception as de:
                            logger.warning(f"Failed to delete intermediate message {msg.id}: {de}")

                # Clear session reference from chatbot's header views cache
                self.chatbot._header_views = {
                    tid: val
                    for tid, val in self.chatbot._header_views.items()
                    if not tid.startswith(self.session_id) and tid != self.session_id
                }

                self.chatbot._turn_special_items.pop(self.session_id, None)
                self.chatbot._turn_special_msg.pop(self.session_id, None)

            await interaction.followup.send(
                content="♻️ Session successfully cleared. The next message will initiate a new session.",
                ephemeral=True,
            )
        except Exception as e:
            logger.error(f"Failed to clear session {self.session_id}: {e}", exc_info=True)
            await interaction.followup.send(
                content=f"⚠️ Failed to clear session: {e}",
                ephemeral=True,
            )

    def _generate_html_trajectory(self, history: list[Message]) -> str:
        """Generate an interactive dark-mode HTML document visualizing the agent trajectory."""
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        timeline_items = []
        for msg in history:
            role_class = "system"
            badge_class = "system"
            emoji_icon = "⚙️"
            label = "System"

            if msg.role == MessageRole.USER:
                role_class = "user"
                badge_class = "user"
                emoji_icon = "👤"
                label = "User"
            elif msg.role == MessageRole.ASSISTANT:
                if msg.type == MessageType.THOUGHT:
                    role_class = "thought"
                    badge_class = "thought"
                    emoji_icon = "💭"
                    label = "Thought"
                else:
                    role_class = "assistant"
                    badge_class = "assistant"
                    emoji_icon = "🤖"
                    label = "Assistant"
            elif msg.role == MessageRole.TOOL:
                if msg.type == MessageType.TOOL_CALL:
                    role_class = "tool-call"
                    badge_class = "tool"
                    emoji_icon = "🛠️"
                    label = "Tool Call"
                else:
                    if msg.metadata.get("tool_error"):
                        role_class = "tool-error"
                        badge_class = "tool"
                        emoji_icon = "📥"
                        label = "Tool Error"
                    else:
                        role_class = "tool-success"
                        badge_class = "tool"
                        emoji_icon = "📥"
                        label = "Tool Result"

            msg_time = datetime.datetime.fromtimestamp(msg.timestamp).strftime("%Y-%m-%d %H:%M:%S")
            escaped_content = html.escape(msg.content)

            is_multiline = "\n" in msg.content or len(msg.content) > 80
            if is_multiline:
                first_line = html.escape(msg.content.splitlines()[0])
                content_html = (
                    f'\n                <div class="entry-summary">{first_line} ...</div>\n'
                    f'                <button class="details-toggle" id="btn-{msg.id}" '
                    f"onclick=\"toggleContent('{msg.id}')\">Expand details</button>\n"
                    f'                <div class="collapsed-content" id="{msg.id}">\n'
                    f"                    <pre><code>{escaped_content}</code></pre>\n"
                    f"                </div>\n"
                )
            else:
                content_html = f'<div class="entry-content">{escaped_content}</div>'

            item_html = f"""
            <div class="entry {role_class}">
                <div class="entry-marker">{emoji_icon}</div>
                <div class="entry-header">
                    <div class="entry-title">
                        <span class="badge {badge_class}">{label}</span>
                        <strong>{html.escape(msg.sender)}</strong>
                    </div>
                    <div class="entry-time">{msg_time}</div>
                </div>
                {content_html}
            </div>
            """
            timeline_items.append(item_html)

        timeline_html = "\n".join(timeline_items)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Trajectory Viewer</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?\
family=Fira+Code:wght@400;500&amp;family=Inter:wght@400;500;600;700&amp;display=swap" \
rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: #334155;
            --accent-thought: #a78bfa;
            --accent-tool-call: #fbbf24;
            --accent-tool-success: #34d399;
            --accent-tool-error: #f87171;
            --accent-system: #64748b;
            --accent-user: #38bdf8;
            --accent-assistant: #ec4899;
        }}

        body {{
            background-color: var(--bg-dark);
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}

        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}

        header {{
            margin-bottom: 32px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
        }}

        h1 {{
            font-size: 28px;
            font-weight: 700;
            margin: 0 0 8px 0;
            background: linear-gradient(135deg, #38bdf8, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .subtitle {{
            color: var(--text-secondary);
            font-size: 14px;
            margin: 0;
        }}

        .timeline {{
            position: relative;
            padding-left: 20px;
            border-left: 2px solid var(--border-color);
        }}

        .entry {{
            position: relative;
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}

        .entry:hover {{
            transform: translateY(-2px);
            border-color: #475569;
        }}

        .entry-marker {{
            position: absolute;
            left: -31px;
            top: 20px;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background-color: var(--bg-dark);
            border: 3px solid var(--accent-system);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10;
            font-size: 12px;
        }}

        .entry.thought {{ border-left: 4px solid var(--accent-thought); }}
        .entry.thought .entry-marker {{ border-color: var(--accent-thought); }}

        .entry.tool-call {{ border-left: 4px solid var(--accent-tool-call); }}
        .entry.tool-call .entry-marker {{ border-color: var(--accent-tool-call); }}

        .entry.tool-success {{ border-left: 4px solid var(--accent-tool-success); }}
        .entry.tool-success .entry-marker {{ border-color: var(--accent-tool-success); }}

        .entry.tool-error {{ border-left: 4px solid var(--accent-tool-error); }}
        .entry.tool-error .entry-marker {{ border-color: var(--accent-tool-error); }}

        .entry.system {{ border-left: 4px solid var(--accent-system); }}
        .entry.system .entry-marker {{ border-color: var(--accent-system); }}

        .entry.user {{ border-left: 4px solid var(--accent-user); }}
        .entry.user .entry-marker {{ border-color: var(--accent-user); }}

        .entry.assistant {{ border-left: 4px solid var(--accent-assistant); }}
        .entry.assistant .entry-marker {{ border-color: var(--accent-assistant); }}

        .entry-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }}

        .entry-title {{
            font-weight: 600;
            font-size: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .entry-time {{
            color: var(--text-secondary);
            font-size: 12px;
        }}

        .entry-summary {{
            font-size: 14px;
            color: var(--text-secondary);
            font-style: italic;
        }}

        .entry-content {{
            font-size: 14px;
            color: #e2e8f0;
            white-space: pre-wrap;
            word-break: break-word;
        }}

        pre {{
            background-color: #090d16;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            border: 1px solid #1e293b;
            font-family: 'Fira Code', monospace;
            font-size: 13px;
            margin-top: 8px;
            margin-bottom: 8px;
        }}

        code {{
            font-family: 'Fira Code', monospace;
            font-size: 13px;
        }}

        .details-toggle {{
            cursor: pointer;
            background: none;
            border: none;
            color: #38bdf8;
            font-size: 13px;
            font-weight: 500;
            padding: 4px 8px;
            border-radius: 4px;
            transition: background-color 0.2s;
            margin-top: 8px;
        }}

        .details-toggle:hover {{
            background-color: rgba(56, 189, 248, 0.1);
        }}

        .collapsed-content {{
            display: none;
        }}

        .badge {{
            font-size: 11px;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 4px;
            text-transform: uppercase;
        }}

        .badge.thought {{ background-color: rgba(167, 139, 250, 0.2); color: var(--accent-thought); }}
        .badge.tool {{ background-color: rgba(251, 191, 36, 0.2); color: var(--accent-tool-call); }}
        .badge.system {{ background-color: rgba(100, 116, 139, 0.2); color: var(--accent-system); }}
        .badge.user {{ background-color: rgba(56, 189, 248, 0.2); color: var(--accent-user); }}
        .badge.assistant {{ background-color: rgba(236, 72, 153, 0.2); color: var(--accent-assistant); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Agent Session Trajectory</h1>
            <p class="subtitle">Session ID: {self.session_id} | Generated at {now_str}</p>
        </header>

        <div class="timeline">
            {timeline_html}
        </div>
    </div>
    <script>
        function toggleContent(id) {{
            const el = document.getElementById(id);
            const btn = document.getElementById('btn-' + id);
            if (el.style.display === 'block') {{
                el.style.display = 'none';
                btn.textContent = 'Expand details';
            }} else {{
                el.style.display = 'block';
                btn.textContent = 'Collapse details';
            }}
        }}
    </script>
</body>
</html>"""


def _get_local_timezone_name() -> str:
    """Retrieve the local system timezone name (e.g., 'Asia/Shanghai')."""
    try:
        return tzlocal.get_localzone().key or "UTC"
    except Exception:
        return datetime.datetime.now().astimezone().tzname() or "UTC"


class QuestionView(discord.ui.View):
    """Dynamic Discord View representing a multiple-choice question.

    Renders choice options as action buttons. Selecting a choice posts a
    simulated user message response directly to the Kesoku gateway.
    """

    def __init__(
        self,
        gateway: Gateway,
        session_id: str,
        chatbot: Any,
        question: str,
        choices: list[str],
    ) -> None:
        """Initialize the QuestionView with multiple-choice buttons.

        Args:
            gateway: The Kesoku Gateway instance.
            session_id: Session ID of the conversation.
            chatbot: Reference to the active Discord chatbot.
            question: The question text block.
            choices: A list of string choice values representing buttons.
        """
        super().__init__(timeout=None)
        self.gateway = gateway
        self.session_id = session_id
        self.chatbot = chatbot
        self.question = question
        self.choices = choices

        for idx, choice in enumerate(choices):
            button = discord.ui.Button(
                style=discord.ButtonStyle.primary,
                label=choice,
                custom_id=f"btn_q_{session_id}_{idx}_{choice[:20]}",
            )
            button.callback = self.make_callback(choice)
            self.add_item(button)

    def make_callback(self, choice: str) -> Any:
        """Create a callback function bound to a specific multiple-choice value.

        Args:
            choice: The string choice value.

        Returns:
            A callback coroutine for the button interaction.
        """

        async def callback(interaction: discord.Interaction) -> None:
            # Defer the interaction response
            await interaction.response.defer()

            # Disable all buttons to prevent multiple clicks or duplicate responses
            for item in self.children:
                if isinstance(item, discord.ui.Button):
                    item.disabled = True

            # Edit the interaction message to show disabled buttons
            await interaction.message.edit(view=self)

            # Send a visible confirmation message to the channel
            response_msg = await interaction.channel.send(f"<@{interaction.user.id}> selected: **{choice}**")

            # Construct and post the user message to the Gateway
            tz_name = _get_local_timezone_name()
            discord_msg_content = (
                f"`{interaction.user.display_name}` <@{interaction.user.id}> "
                f"at `{response_msg.created_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')} {tz_name}`:\n"
                f"{choice}"
            )
            msg = Message(
                session_id=self.session_id,
                chatbot_id=self.chatbot.chatbot_id,
                channel_id=str(interaction.channel_id),
                sender=interaction.user.display_name,
                role=MessageRole.USER,
                type=MessageType.TEXT,
                content=discord_msg_content,
                timestamp=response_msg.created_at.timestamp(),
                status=MessageStatus.PENDING_AGENT,
                metadata={
                    "discord_message_id": str(response_msg.id),
                    "discord_author_id": str(interaction.user.id),
                },
            )
            await self.gateway.post(msg)

            # Trigger typing task since a new user message was posted
            channel_id_str = str(interaction.channel_id)
            if channel_id_str not in self.chatbot._typing_tasks:
                self.chatbot._typing_tasks[channel_id_str] = asyncio.create_task(
                    self.chatbot._keep_typing(interaction.channel)
                )

        return callback
