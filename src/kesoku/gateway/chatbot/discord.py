"""Discord chatbot adapter for Kesoku AI Agent framework.

Connects Discord channels and threads with Kesoku Gateway using Pub/Sub.
"""

import asyncio
import datetime
import html
import io
import os

import discord
import tzlocal

from kesoku.agent.prompt import build_sys_prompt
from kesoku.config import get_config
from kesoku.constants import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    STATUS_DELIVERED,
    STATUS_PENDING_AGENT,
    TYPE_TEXT,
    TYPE_THOUGHT,
    TYPE_TOOL_CALL,
)
from kesoku.db import Message
from kesoku.gateway.chatbot.base import Chatbot, parse_message_content
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


def _get_local_timezone_name() -> str:
    """Retrieve the local system timezone name (e.g., 'Asia/Shanghai')."""
    try:
        return tzlocal.get_localzone().key or "UTC"
    except Exception:
        return datetime.datetime.now().astimezone().tzname() or "UTC"


def _build_discord_sys_prompt(
    channel: discord.Thread | discord.DMChannel | discord.GroupChannel | discord.TextChannel,
    author: discord.User | discord.Member,
) -> str:
    """Build the default system prompt for a Discord thread session.

    Args:
        channel: The Discord channel or thread instance.
        author: The author of the initiating message.

    Returns:
        The built system prompt string.
    """
    is_dm = getattr(channel, "guild", None) is None

    # Build user member lines (only for group chats/servers, not DMs)
    members_section = ""
    if not is_dm:
        member_lines = []
        if hasattr(channel, "guild") and channel.guild and hasattr(channel.guild, "members"):
            for m in channel.guild.members:
                if not m.bot:
                    member_lines.append(f"- {m.display_name} (ID: {m.id})")
        if not member_lines:
            member_lines.append(f"- {author.display_name} (ID: {author.id})")
        members_str = "\n".join(member_lines)
        members_section = f"\n## Users on the server\n\n{members_str}\n"

    time_section = """\n## Time
The user message contains their user id and the **real** time that the message is sent.
The time is very important to prevent your hallucination about the world status.\n"""

    # Build location instruction and channel topic
    if is_dm:
        location_instruction = "You are talking to the user via discord."
        topic_section = ""
    else:
        guild_name = channel.guild.name if channel.guild else "Unknown Server"
        if isinstance(channel, discord.Thread):
            thread_name = channel.name
            thread_id = channel.id
            parent = channel.parent
            channel_name = parent.name if parent else "unknown-channel"
            channel_id = parent.id if parent else "unknown"
            topic = getattr(parent, "topic", None) or ""
            location_instruction = (
                f"You are currently chatting in a Discord thread named \"#{thread_name}\" (ID: {thread_id}) "
                f"under channel \"#{channel_name}\" (ID: {channel_id}) on the server '{guild_name}'."
            )
        else:
            channel_name = channel.name
            channel_id = channel.id
            topic = getattr(channel, "topic", None) or ""
            location_instruction = (
                f"You are currently chatting in a Discord channel named \"#{channel_name}\" (ID: {channel_id}) "
                f"on the server '{guild_name}'."
            )

        topic_section = f"## Channel Topic\n{topic}" if topic else ""

    mention_section = ""
    if not is_dm:
        mention_section = (
            "\n## Mentioning Users\n"
            "When mentioning or referring to a user, use Discord tag syntax <@USER_ID>.\n"
        )

    format_section = """
## Response Format
This format requirement only applies for your response in discord (not for writing files etc).
- Discord doesn't support latex syntax for math, so use plain text or emojis when you want to
show math. e.g. use "exp(x)" instead of "$e^x$", use "∞" instead of "$\\inf$".
- Discord doesn't support level 4+ headings, so use level 3 headings at most (Start with level 1 heading).
    """

    discord_prompt = f"""
# Discord Instructions
{location_instruction}
{members_section}
{mention_section}
{time_section}
{topic_section}
{format_section}
    """
    return build_sys_prompt(discord_prompt)


class MessageHeaderView(discord.ui.View):
    """Persistent Discord view representing the conversation header with interactive trajectory viewer."""

    def __init__(self, gateway: Gateway, session_id: str) -> None:
        """Initialize the MessageHeaderView.

        Args:
            gateway: The Kesoku Gateway instance.
            session_id: Session ID of the conversation.
        """
        super().__init__(timeout=None)
        self.gateway = gateway
        self.session_id = session_id

    @discord.ui.button(
        label="View Trajectory",
        style=discord.ButtonStyle.secondary,
        emoji="📜",
        custom_id="btn_view_trajectory",
    )
    async def view_trajectory(self, interaction: discord.Interaction, button: discord.ui.Button) -> None:
        """Callback triggered when 'View Trajectory' button is clicked."""
        await interaction.response.defer(ephemeral=True)

        try:
            # Fetch entire historical context up to 200 messages in grouped user-facing order
            history = await self.gateway.get_session_history(self.session_id, limit=200, order="grouped")

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

    def _generate_html_trajectory(self, history: list[Message]) -> str:
        """Generate an interactive dark-mode HTML document visualizing the agent trajectory."""
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        timeline_items = []
        for msg in history:
            role_class = "system"
            badge_class = "system"
            emoji_icon = "⚙️"
            label = "System"

            if msg.role == ROLE_USER:
                role_class = "user"
                badge_class = "user"
                emoji_icon = "👤"
                label = "User"
            elif msg.role == ROLE_ASSISTANT:
                if msg.type == TYPE_THOUGHT:
                    role_class = "thought"
                    badge_class = "thought"
                    emoji_icon = "💭"
                    label = "Thought"
                else:
                    role_class = "assistant"
                    badge_class = "assistant"
                    emoji_icon = "🤖"
                    label = "Assistant"
            elif msg.role == ROLE_TOOL:
                if msg.type == TYPE_TOOL_CALL:
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
                    f'onclick="toggleContent(\'{msg.id}\')">Expand details</button>\n'
                    f'                <div class="collapsed-content" id="{msg.id}">\n'
                    f'                    <pre><code>{escaped_content}</code></pre>\n'
                    f'                </div>\n'
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


class DiscordChatbot(Chatbot):
    """Discord chatbot adapter connecting to Kesoku Gateway broker."""

    def __init__(self, chatbot_id: str, gateway: Gateway, bot_token: str | None = None) -> None:
        """Initialize the Discord chatbot adapter.

        Args:
            chatbot_id: Unique identifier for this chatbot instance.
            gateway: Kesoku Gateway instance.
            bot_token: Optional Discord bot token. Defaults to config setting.
        """
        super().__init__(chatbot_id, gateway)
        self.config = get_config()
        self.bot_token = bot_token or self.config.discord.bot_token
        if not self.bot_token:
            raise ValueError("Discord bot token is required but not configured.")

        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.guilds = True
        self.bot = discord.Client(intents=intents)
        self.bot.event(self.on_ready)
        self.bot.event(self.on_message)
        self._subscriber_task: asyncio.Task[None] | None = None
        self._sent_tool_calls: dict[str, discord.Message] = {}
        self._turns_with_header: set[str] = set()
        self._typing_tasks: dict[str, asyncio.Task[None]] = {}

    async def start(self) -> None:
        """Start the Discord bot and Gateway listener subscriber background loop."""
        self._subscriber_task = asyncio.create_task(super().start())
        logger.info(f"Connecting Discord bot '{self.chatbot_id}'...")
        await self.bot.start(self.bot_token)

    def stop(self) -> None:
        """Stop the Discord bot and subscriber listener task."""
        if self._subscriber_task and not self._subscriber_task.done():
            self._subscriber_task.cancel()
        super().stop()
        for task in self._typing_tasks.values():
            task.cancel()
        self._typing_tasks.clear()
        if not self.bot.is_closed():
            asyncio.create_task(self.bot.close())

    async def _keep_typing(
        self, channel: discord.Thread | discord.DMChannel | discord.GroupChannel | discord.TextChannel
    ) -> None:
        """Keep sending typing status to Discord channel/thread in a loop.

        Automatically times out after 10 minutes to prevent infinite typing.
        """
        try:
            async with channel.typing():
                await asyncio.sleep(600)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Failed to send typing status in channel {channel.id}: {e}")
        finally:
            self._typing_tasks.pop(str(channel.id), None)


    async def on_ready(self) -> None:
        """Callback invoked when Discord bot successfully connects and logs in."""
        logger.info(f"Discord chatbot '{self.chatbot_id}' successfully logged in as {self.bot.user}")

    async def on_message(self, message: discord.Message) -> None:
        """Process incoming messages from Discord and ingest into Kesoku Gateway.

        Args:
            message: Incoming discord.Message instance.
        """
        # Ignore bot's own messages
        if message.author == self.bot.user:
            return

        # Check user_allowlist filtering
        allowlist = self.config.discord.user_allowlist
        if allowlist:
            is_allowed = (
                str(message.author.id) in allowlist
                or message.author.name in allowlist
                or message.author.display_name in allowlist
            )
            if not is_allowed:
                # Unlisted users only trigger replies if explicitly mentioning the bot
                if self.bot.user not in message.mentions:
                    return

        # Check if message explicitly mentions someone else but NOT this bot
        other_mentioned = [u for u in message.mentions if u != self.bot.user]
        if other_mentioned and self.bot.user not in message.mentions:
            return

        # Thread-based context separation
        thread: discord.Thread | None = None
        if isinstance(message.channel, discord.Thread):
            thread = message.channel
        else:
            # In regular channel; find or create a thread
            if hasattr(message.channel, "guild") and message.channel.guild:
                thread = message.channel.guild.get_thread(message.id)
            if not thread and hasattr(message, "thread") and message.thread:
                thread = message.thread

            if not thread:
                try:
                    title = message.content[:30] + ("..." if len(message.content) > 30 else "")
                    if not title.strip():
                        title = f"Chat with {message.author.display_name}"
                    thread = await message.create_thread(name=title)
                except discord.HTTPException as e:
                    logger.warning(
                        f"Failed to create thread on message {message.id} (concurrent bot creation?): {e}"
                    )
                    await asyncio.sleep(0.5)
                    if hasattr(message.channel, "guild") and message.channel.guild:
                        thread = message.channel.guild.get_thread(message.id)
                    if not thread and hasattr(message.channel, "archived_threads"):
                        async for t in message.channel.archived_threads(limit=10):
                            if t.id == message.id:
                                thread = t
                                break
            if not thread:
                logger.error(f"Could not create or retrieve thread for message {message.id}")
                return

            try:
                await thread.join()
            except Exception as je:
                logger.debug(f"Already joined or error joining thread {thread.id}: {je}")

        channel_id = str(thread.id)
        session = await self.gateway.get_session_by_channel(self.chatbot_id, channel_id)

        if not session:
            sys_prompt = _build_discord_sys_prompt(thread, message.author)
            session = await self.gateway.create_session(
                title=thread.name,
                system_prompt=sys_prompt,
                created_at=message.created_at.timestamp(),
            )
            session_id = session.id
        else:
            session_id = session.id
            await self.gateway.update_session_updated_at(session_id)

        # Ingest user message into Gateway
        tz_name = _get_local_timezone_name()
        discord_msg_content = (
            f"`{message.author.display_name}` <@{message.author.id}> "
            f"at `{message.created_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')} {tz_name}`:\n"
            f"{message.content}"
        )
        msg = Message(
            session_id=session_id,
            chatbot_id=self.chatbot_id,
            channel_id=channel_id,
            sender=message.author.display_name,
            role=ROLE_USER,
            type=TYPE_TEXT,
            content=discord_msg_content,
            timestamp=message.created_at.timestamp(),
            status=STATUS_PENDING_AGENT,
            metadata={"discord_message_id": str(message.id), "discord_author_id": str(message.author.id)},
        )
        await self.gateway.post(msg)

        # Trigger typing status for the thread/channel while agent is thinking
        if channel_id not in self._typing_tasks:
            self._typing_tasks[channel_id] = asyncio.create_task(self._keep_typing(thread))


    async def handle_message(self, message: Message) -> None:
        """Process outgoing message from Gateway and send to target Discord thread.

        Args:
            message: Outgoing Message instance.
        """
        try:
            target_id = int(message.channel_id)
        except ValueError:
            logger.error(f"Invalid Discord channel_id: {message.channel_id}")
            return

        channel = self.bot.get_channel(target_id)
        if not channel:
            try:
                channel = await self.bot.fetch_channel(target_id)  # type: ignore
            except Exception as fe:
                logger.error(f"Failed to fetch Discord channel {target_id}: {fe}")
                return

        is_special_message = False
        output_text = ""

        if message.role == ROLE_ASSISTANT:
            if message.type == TYPE_THOUGHT:
                is_special_message = True
                first_line = message.content.split("\n")[0].strip()
                hidden_chars = len(message.content) - len(first_line)
                output_text = f"💭 *Thought Process:* {first_line} ... *(+{hidden_chars} chars)*"
            else:
                output_text = message.content
        elif message.role == ROLE_TOOL:
            is_special_message = True
            tool_name = message.metadata.get("tool_name") or message.sender or "unknown_tool"
            if message.type == TYPE_TOOL_CALL:
                output_text = f"🛠️ **{tool_name}** ⏳"
            else:
                if message.metadata.get("tool_error"):
                    output_text = f"📥 **{tool_name}** ❌"
                else:
                    output_text = f"📥 **{tool_name}** ✅"
        elif message.role == ROLE_SYSTEM:
            is_special_message = True
            first_line = message.content.split("\n")[0].strip()
            hidden_chars = len(message.content) - len(first_line)
            output_text = f"⚙️ *System Message:* {first_line} ... *(+{hidden_chars} chars)*"
        else:
            output_text = message.content

        if not output_text:
            output_text = message.content

        # Try in-place editing if it is a tool result
        if message.role == ROLE_TOOL and message.type != TYPE_TOOL_CALL:
            tool_call_msg_id = message.parent_id
            if tool_call_msg_id and tool_call_msg_id in self._sent_tool_calls:
                discord_msg = self._sent_tool_calls.pop(tool_call_msg_id)
                try:
                    await discord_msg.edit(content=output_text)
                    await self.gateway.update_message_status(message.id, STATUS_DELIVERED)
                    return
                except Exception as ee:
                    logger.warning(
                        f"Failed to edit tool call message in-place: {ee}. "
                        "Falling back to sending new message."
                    )

        # Send the MessageHeaderView at the start of the turn if it's a special message
        if is_special_message:
            turn_id = message.parent_id or message.session_id
            if turn_id not in self._turns_with_header:
                try:
                    header_view = MessageHeaderView(self.gateway, message.session_id)
                    await channel.send(view=header_view)
                    self._turns_with_header.add(turn_id)
                except Exception as he:
                    logger.warning(f"Failed to send message header: {he}")

        # Parse output text into segments (handling [file: <path>] blocks)
        segments = parse_message_content(output_text)

        for segment in segments:
            if segment["type"] == "text":
                text_content = segment["content"]
                # Only send if the text chunk is not empty and contains non-whitespace characters
                if text_content.strip():
                    # Newline Chunking (<= 2000 chars) for Discord compatibility
                    lines = text_content.splitlines(keepends=True)
                    current_chunk = ""
                    for line in lines:
                        if len(line) > 2000:
                            if current_chunk:
                                sent_msg = await channel.send(current_chunk)
                                current_chunk = ""
                            for i in range(0, len(line), 2000):
                                sent_msg = await channel.send(line[i : i + 2000])
                        elif len(current_chunk) + len(line) > 2000:
                            sent_msg = await channel.send(current_chunk)
                            current_chunk = line
                        else:
                            current_chunk += line

                    if current_chunk and current_chunk.strip():
                        sent_msg = await channel.send(current_chunk)

                        # Cache the sent message object if it's a tool call for future editing
                        if message.role == ROLE_TOOL and message.type == TYPE_TOOL_CALL:
                            self._sent_tool_calls[message.id] = sent_msg
            elif segment["type"] == "file":
                file_path = segment["path"]
                if not os.path.exists(file_path):  # noqa: ASYNC240
                    logger.error(f"File not found: {file_path}")
                    await channel.send(f"⚠️ File not found: {file_path}")
                else:
                    try:
                        discord_file = discord.File(file_path)
                        await channel.send(file=discord_file)
                    except Exception as e:
                        logger.error(f"Failed to send file {file_path} to Discord: {e}", exc_info=True)
                        await channel.send(f"⚠️ Failed to send file {file_path}: {e}")

        await self.gateway.update_message_status(message.id, STATUS_DELIVERED)

        # Stop typing status when the final assistant response is successfully delivered
        if message.role == ROLE_ASSISTANT and message.type == TYPE_TEXT:
            task = self._typing_tasks.pop(message.channel_id, None)
            if task:
                task.cancel()

