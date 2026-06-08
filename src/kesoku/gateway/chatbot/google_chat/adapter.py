"""Google Chat chatbot adapter for Kesoku AI Agent framework.

Connects Google Chat spaces with the internal Kesoku Gateway using
a Google Cloud Pub/Sub Pull Subscription and GCP public APIs.
"""

import asyncio
import json
import random
import re
import time
from typing import Any

import google.auth
from google.auth import impersonated_credentials
from google.cloud import pubsub_v1
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError

from kesoku.config import get_config
from kesoku.constants import MessageRole, MessageStatus, MessageType
from kesoku.db import Message
from kesoku.gateway.chatbot.base import Chatbot, InboundMessageDTO, parse_message_content
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger

from .cards import GoogleChatCardBuilder

logger = setup_logger(__name__)


def parse_emoji_sequence(emoji_str: str) -> list[str]:
    """Parse a string of emojis into a list of individual emojis.

    Supports space-separated, comma-separated, or a sequence of Unicode emojis.

    Args:
        emoji_str: The configuration string containing emojis.

    Returns:
        A list of individual emoji strings.
    """
    if not emoji_str:
        return []
    if " " in emoji_str:
        return [e.strip() for e in emoji_str.split(" ") if e.strip()]
    if "," in emoji_str:
        return [e.strip() for e in emoji_str.split(",") if e.strip()]

    # Match custom emojis (like :temu:) or any single unicode code point
    tokens = re.findall(r":[a-zA-Z0-9_\-+]+:|.", emoji_str)

    emojis = []
    current: list[str] = []
    for token in tokens:
        if token.startswith(":") and token.endswith(":"):
            if current:
                emojis.append("".join(current))
                current = []
            emojis.append(token)
        else:
            char = token
            # Variation Selector (0xfe0f) or ZWJ (0x200d) or Fitzpatrick skin tones
            if ord(char) in (0xFE0F, 0x200D) or (current and ord(char) in range(0x1F3FB, 0x1F3FF + 1)):
                current.append(char)
            else:
                if current:
                    emojis.append("".join(current))
                current = [char]
    if current:
        emojis.append("".join(current))
    return emojis


class GoogleChatChatbot(Chatbot):
    """Chatbot adapter connecting Google Chat spaces with Kesoku Gateway via Pub/Sub."""

    def __init__(self, chatbot_id: str, gateway: Gateway) -> None:
        """Initialize the Google Chat chatbot adapter.

        Args:
            chatbot_id: Unique identifier for this chatbot instance.
            gateway: The Kesoku Gateway instance managing routing and persistence.

        Raises:
            ValueError: If critical configuration settings are missing.
        """
        super().__init__(chatbot_id, gateway)
        cfg = get_config()
        self.config = cfg.google_chat

        if not self.config.enabled:
            raise ValueError("Google Chat chatbot is disabled in configuration.")

        if not self.config.project_id or not self.config.topic_id or not self.config.subscription_id:
            raise ValueError("Google Chat is enabled but project_id, topic_id, or subscription_id are not configured.")

        self._running = False
        self._pubsub_task: asyncio.Task[None] | None = None
        self._foldable_ui_messages: dict[str, dict[str, Any]] = {}
        self._active_user_message_names: dict[str, str] = {}
        self._used_reactions: dict[str, dict[str, str]] = {}

        # Load credentials and initialize Google APIs
        self._credentials, self._project_id = self._load_credentials()
        self._subscriber_client = pubsub_v1.SubscriberClient(credentials=self._credentials)
        self._subscription_path = self._subscriber_client.subscription_path(
            self.config.project_id, self.config.subscription_id
        )

        # Build standard Google Chat API client
        self._chat_service: Resource = build("chat", "v1", credentials=self._credentials)

        # Build user authenticated Google Chat API client if emoji reactions are enabled
        self._user_chat_service: Resource | None = None
        if self.config.reaction_emoji:
            try:
                user_creds, _ = self._load_user_credentials()
                self._user_chat_service = build("chat", "v1", credentials=user_creds)
            except Exception as e:
                logger.warning(
                    f"Failed to load user credentials or build user_chat_service for reactions: {e}."
                    " Reaction emoji feature will be disabled."
                )

    def _load_credentials(self) -> tuple[Any, str]:
        """Load GCP credentials using ADC, key files, or impersonation options.

        Returns:
            A tuple of (credentials, project_id).
        """
        scopes = [
            "https://www.googleapis.com/auth/pubsub",
            "https://www.googleapis.com/auth/chat.bot",
        ]

        # Option 1: Explicit Service Account Key File
        if self.config.credentials_json:
            logger.info(f"Loading Google Chat credentials from file: {self.config.credentials_json}")
            return google.auth.load_credentials_from_file(self.config.credentials_json, scopes=scopes)

        # Option 2: Application Default Credentials (ADC)
        source_creds, project_id = google.auth.default(scopes=scopes)

        # Option 3: Explicit Service Account Impersonation (Key-Less Workaround)
        if self.config.impersonate_service_account:
            logger.info(f"Impersonating Google Chat service account: {self.config.impersonate_service_account}")
            impersonated_creds = impersonated_credentials.Credentials(
                source_credentials=source_creds,
                target_principal=self.config.impersonate_service_account,
                target_scopes=scopes,
                lifetime=3600,
            )
            # Use config project_id as fallback
            return impersonated_creds, project_id or self.config.project_id

        logger.info("Loading Google Chat credentials using Application Default Credentials (ADC).")
        return source_creds, project_id or self.config.project_id

    def _load_user_credentials(self) -> tuple[Any, str]:
        """Load GCP user credentials strictly using ADC (no service account or impersonation).

        Returns:
            A tuple of (credentials, project_id).
        """
        scopes = [
            "https://www.googleapis.com/auth/chat.messages.reactions.create",
            "https://www.googleapis.com/auth/chat.messages",
        ]
        logger.info("Loading Google Chat user credentials strictly using ADC.")
        source_creds, project_id = google.auth.default(scopes=scopes)
        return source_creds, project_id or (self.config.project_id if self.config.project_id else "")

    async def start(self) -> None:
        """Start listening to outgoing agent responses and run the Pub/Sub Pull task."""
        self._running = True
        # Start base subscriber listener for outgoing agent responses
        self._listener_task = asyncio.create_task(super().start())
        # Start background subscriber loop for incoming Google Chat events
        self._pubsub_task = asyncio.create_task(self._run_pubsub_pull())

        logger.info(f"Google Chat chatbot '{self.chatbot_id}' started on subscription: {self._subscription_path}")
        await asyncio.gather(self._listener_task, self._pubsub_task)

    def stop(self) -> None:
        """Stop the chatbot listeners cleanly."""
        self._running = False
        super().stop()
        if self._pubsub_task and not self._pubsub_task.done():
            self._pubsub_task.cancel()
        logger.info(f"Google Chat chatbot '{self.chatbot_id}' stopped.")

    async def _run_pubsub_pull(self) -> None:
        """Asynchronously pull and process interaction events from Google Cloud Pub/Sub."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[pubsub_v1.subscriber.message.Message] = asyncio.Queue()

        def _pubsub_callback(pubsub_msg: pubsub_v1.subscriber.message.Message) -> None:
            """Thread-safe callback pushing incoming messages into the asyncio queue."""
            loop.call_soon_threadsafe(queue.put_nowait, pubsub_msg)

        # Start standard pull subscription
        streaming_pull_future = self._subscriber_client.subscribe(self._subscription_path, callback=_pubsub_callback)

        try:
            while self._running:
                pubsub_msg = await queue.get()
                try:
                    await self._on_pubsub_message(pubsub_msg)
                except Exception as e:
                    logger.error(f"Error processing incoming Pub/Sub message: {e}", exc_info=True)
                finally:
                    # Always acknowledge the message to remove it from Pub/Sub queue
                    pubsub_msg.ack()
        except asyncio.CancelledError:
            streaming_pull_future.cancel()
            logger.debug("Google Chat Pub/Sub pull task cancelled.")

    async def _on_pubsub_message(self, pubsub_msg: pubsub_v1.subscriber.message.Message) -> None:
        """Parse and route a single Pub/Sub message payload.

        Args:
            pubsub_msg: The Pub/Sub Message instance from GCP.
        """
        payload_bytes = pubsub_msg.data
        if not payload_bytes:
            return

        event = json.loads(payload_bytes.decode("utf-8"))
        logger.debug(f"Incoming event raw payload: {json.dumps(event)}")

        # Normalize Google Workspace Add-on / preprod wrapped payloads
        if "chat" in event:
            chat_data = event["chat"]
            msg_payload = chat_data.get("messagePayload", {})
            if msg_payload:
                # Construct a normalized MESSAGE event structure
                normalized_event = {
                    "type": "MESSAGE",
                    "space": msg_payload.get("space", {}),
                    "message": msg_payload.get("message", {}),
                }
                await self._handle_incoming_message(normalized_event)
            return

        event_type = event.get("type")
        logger.debug(f"Google Chat received Pub/Sub event type: {event_type}")

        # Direct / Space Messages
        if event_type == "MESSAGE":
            await self._handle_incoming_message(event)

    async def pre_ingest_interruption_hook(self, session: Any, dto: InboundMessageDTO) -> None:
        """Hook executed after session is resolved/created, but before posting."""
        session_id = session.id
        channel_id = dto.channel_id

        foldable = self._foldable_ui_messages.pop(session_id, None)
        if foldable and foldable["name"]:
            history = await self.gateway.db.get_session_history(session_id, limit=20)
            prev_user_msg = None
            for msg in reversed(history):
                if msg.role == MessageRole.USER:
                    prev_user_msg = msg
                    break
            metrics = prev_user_msg.metadata.get("turn_metrics") if prev_user_msg else None

            body = {
                "cardsV2": [
                    GoogleChatCardBuilder.build_foldable_ui_card(
                        session_id,
                        foldable["items"],
                        status="interrupted",
                        metrics=metrics,
                    )
                ]
            }
            if channel_id and "threads" in channel_id:
                body["thread"] = {"name": channel_id}
            try:
                await asyncio.to_thread(
                    self._chat_service.spaces()
                    .messages()
                    .patch(
                        name=foldable["name"],
                        body=body,
                        updateMask="cardsV2",
                    )
                    .execute
                )
            except Exception as e:
                logger.error(f"Failed to finalize previous Google Chat foldable UI card on thought interruption: {e}")

    async def post_ingest_hook(self, session: Any, message: Message, dto: InboundMessageDTO) -> None:
        """Hook executed after the message is successfully posted to the gateway."""
        session_id = session.id
        message_name = dto.raw_metadata.get("message_name")
        if not message_name:
            return

        # Store active user message name for this session
        self._active_user_message_names[session_id] = message_name
        self._used_reactions[message_name] = {}

        # Optionally add a random emoji reaction if configured and using user credentials
        if self.config.reaction_emoji and self._user_chat_service:
            emojis = parse_emoji_sequence(self.config.reaction_emoji)
            if emojis:
                selected_emoji = random.choice(emojis)
                asyncio.create_task(self._add_reaction(message_name, selected_emoji))

    async def _handle_incoming_message(self, event: dict[str, Any]) -> None:
        """Process an incoming standard text message or mention.

        Args:
            event: The decoded JSON interaction event payload.
        """
        message_data = event.get("message", {})
        sender_data = message_data.get("sender", {})
        space_data = event.get("space", {})
        thread_data = message_data.get("thread", {})

        sender_name = sender_data.get("displayName", "User")
        sender_id = sender_data.get("name", "users/unknown")
        text = message_data.get("argumentText", "")
        if not text.strip():
            text = message_data.get("text", "")
        text = text.strip()
        space_name = space_data.get("name")  # e.g., "spaces/AAAAxxxx"
        thread_name = thread_data.get("name")  # e.g., "spaces/AAAAxxxx/threads/YYYY"

        sender_email = sender_data.get("email") or ""

        # Security: Filter via user allowlist if configured
        if self.config.user_allowlist:
            if sender_id not in self.config.user_allowlist and sender_email not in self.config.user_allowlist:
                logger.warning(f"Google Chat: Ignoring unauthorized sender {sender_name} ({sender_id}/{sender_email})")
                return

        if not text:
            return

        # Define the thread context as the logical channel ID
        channel_id = thread_name if thread_name else space_name

        # Compile prompt
        custom_prompt = self._build_gchat_custom_prompt(space_data, sender_name)

        # Resolve channel role
        db_role = await self.gateway.db.get_channel_role(self.chatbot_id, channel_id)
        role = db_role if isinstance(db_role, str) else "default"

        dto = InboundMessageDTO(
            sender_id=sender_name,
            channel_id=channel_id,
            text=text,
            message_id=message_data.get("name", ""),
            timestamp=time.time(),
            raw_metadata={
                "message_name": message_data.get("name"),
                "google_chat_sender_email": sender_email,
                "sender_name": f"{sender_name} (Email: {sender_email})" if sender_email else sender_name,
            },
            session_title=f"Google Chat Session: {text[:30]}",
            custom_prompt=custom_prompt,
            role=role,
        )

        async def reply_func(reply_text: str) -> None:
            body = {"text": reply_text}
            if channel_id and "threads" in channel_id:
                body["thread"] = {"name": channel_id}

            parent_space = channel_id.split("/threads/")[0] if channel_id else "spaces/unknown"
            try:
                await asyncio.to_thread(
                    self._chat_service.spaces()
                    .messages()
                    .create(
                        parent=parent_space,
                        body=body,
                        messageReplyOption="REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD",
                    )
                    .execute
                )
            except Exception as e:
                logger.error(f"Google Chat: failed to send command reply: {e}", exc_info=True)

        await self.ingest_message(dto, raw_message=event, reply_callback=reply_func)

    async def _add_reaction(self, message_name: str, emoji: str) -> None:
        """Add an emoji reaction to a Google Chat message using user credentials.

        If the emoji already exists as a reaction, removes it instead.

        Args:
            message_name: The fully qualified resource name of the message.
            emoji: The emoji string to react with.
        """
        if not self._user_chat_service:
            return

        # Initialize used reactions dictionary for the message if not present
        if message_name not in self._used_reactions:
            self._used_reactions[message_name] = {}

        used_map = self._used_reactions[message_name]

        # Toggle Deletion: If this emoji is already in the used map, delete it!
        if emoji in used_map:
            reaction_name = used_map[emoji]
            try:
                await asyncio.to_thread(
                    self._user_chat_service.spaces().messages().reactions().delete(name=reaction_name).execute
                )
                used_map.pop(emoji)
                logger.debug(f"Successfully removed reaction '{emoji}' from message: {message_name}")
            except Exception as e:
                logger.error(f"Failed to delete reaction '{emoji}' from message {message_name}: {e}")
            return

        emoji_payload: dict[str, Any] = {}
        if emoji.startswith(":") and emoji.endswith(":"):
            emoji_payload = {"customEmoji": {"uid": emoji}}
        else:
            emoji_payload = {"unicode": emoji}

        body = {"emoji": emoji_payload}
        try:
            res = await asyncio.to_thread(
                self._user_chat_service.spaces().messages().reactions().create(parent=message_name, body=body).execute
            )
            # Store the created reaction resource name mapped to the emoji
            used_map[emoji] = res["name"]
            logger.debug(f"Successfully reacted with '{emoji}' to message: {message_name}")
        except HttpError as e:
            if e.resp.status == 409:
                # Duplicate reaction reported by the API. We don't have the name in memory,
                # but since it already exists, let's query reactions to find it and delete it.
                logger.warning(
                    f"Reaction '{emoji}' already exists on {message_name} (Http 409). Resolving and deleting."
                )
                try:
                    reactions_list_res = await asyncio.to_thread(
                        self._user_chat_service.spaces().messages().reactions().list(parent=message_name).execute
                    )
                    reactions_list = reactions_list_res.get("reactions", [])
                    # Find our reaction
                    found = False
                    for r in reactions_list:
                        r_emoji = r.get("emoji", {})
                        r_unicode = r_emoji.get("unicode")
                        r_custom = r_emoji.get("customEmoji", {}).get("uid")
                        if r_unicode == emoji or r_custom == emoji:
                            # Found it! Let's delete it
                            await asyncio.to_thread(
                                self._user_chat_service.spaces().messages().reactions().delete(name=r["name"]).execute
                            )
                            logger.debug(
                                f"Successfully removed duplicate reaction '{emoji}' from message: {message_name}"
                            )
                            found = True
                            break
                    if not found:
                        logger.warning(f"Could not resolve duplicate reaction '{emoji}' to delete on {message_name}.")
                except Exception as delete_error:
                    logger.error(
                        f"Failed to resolve/delete duplicate reaction '{emoji}' on {message_name}: {delete_error}"
                    )
            else:
                logger.error(f"Google Chat API error when adding reaction '{emoji}': {e}")
        except Exception as e:
            logger.error(f"Failed to add reaction '{emoji}' to message {message_name}: {e}")

    async def handle_message(self, message: Message) -> None:
        """Process and send outgoing assistant messages to the Google Chat API.

        Args:
            message: The Message instance to handle.
        """
        # Only handle messages destined for this chatbot adapter
        if message.chatbot_id != self.chatbot_id:
            return

        logger.debug(f"Google Chat processing outgoing message ID {message.id}")

        session_id = message.session_id
        is_intermediate = (
            (message.role == MessageRole.ASSISTANT and message.type == MessageType.THOUGHT)
            or (message.role == MessageRole.TOOL)
            or (message.role == MessageRole.SYSTEM)
        )

        if is_intermediate:
            foldable = self._foldable_ui_messages.get(session_id)
            if not foldable:
                foldable = {
                    "name": None,
                    "items": [],
                }
                self._foldable_ui_messages[session_id] = foldable

            items = foldable["items"]

            if message.role == MessageRole.ASSISTANT and message.type == MessageType.THOUGHT:
                items.append(
                    {
                        "type": "thought",
                        "content": message.content,
                    }
                )
            elif message.role == MessageRole.TOOL and message.type == MessageType.TOOL_CALL:
                tool_name = message.metadata.get("tool_name") or message.sender or "unknown_tool"
                arg_suffix = GoogleChatCardBuilder.get_tool_arguments_suffix(message.metadata.get("tool_arguments"))
                items.append(
                    {
                        "type": "tool_call",
                        "id": message.id,
                        "tool_name": tool_name,
                        "arg_suffix": arg_suffix,
                        "status": "⏳",
                    }
                )

                # Trigger a random emoji reaction for the tool call if configured
                if self.config.reaction_emoji and self._user_chat_service:
                    message_name = self._active_user_message_names.get(session_id)
                    if message_name:
                        emojis = parse_emoji_sequence(self.config.reaction_emoji)
                        if emojis:
                            selected_emoji = random.choice(emojis)
                            asyncio.create_task(self._add_reaction(message_name, selected_emoji))
            elif message.role == MessageRole.TOOL and message.type != MessageType.TOOL_CALL:
                tool_call_msg_id = message.parent_id
                found = False
                if tool_call_msg_id:
                    for item in items:
                        if item.get("type") == "tool_call" and item.get("id") == tool_call_msg_id:
                            if message.metadata.get("tool_error"):
                                item["status"] = "❌"
                            else:
                                item["status"] = "✅"
                            found = True
                            break
                if not found:
                    for item in reversed(items):
                        if item.get("type") == "tool_call" and item.get("status") == "⏳":
                            if message.metadata.get("tool_error"):
                                item["status"] = "❌"
                            else:
                                item["status"] = "✅"
                            break
            elif message.role == MessageRole.SYSTEM:
                items.append(
                    {
                        "type": "system",
                        "content": message.content,
                    }
                )

            # Build the card payload
            body = {"cardsV2": [GoogleChatCardBuilder.build_foldable_ui_card(session_id, items, status="running")]}
            if message.channel_id and "threads" in message.channel_id:
                body["thread"] = {"name": message.channel_id}

            parent_space = message.channel_id.split("/threads/")[0] if message.channel_id else "spaces/unknown"
            try:
                if foldable["name"] is None:
                    res = await asyncio.to_thread(
                        self._chat_service.spaces()
                        .messages()
                        .create(
                            parent=parent_space,
                            body=body,
                            messageReplyOption="REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD",
                        )
                        .execute
                    )
                    foldable["name"] = res["name"]
                else:
                    await asyncio.to_thread(
                        self._chat_service.spaces()
                        .messages()
                        .patch(
                            name=foldable["name"],
                            body=body,
                            updateMask="cardsV2",
                        )
                        .execute
                    )
                await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)
            except Exception as e:
                logger.error(f"Failed to send/update Google Chat foldable UI card: {e}", exc_info=True)
            return

        # Handle final text reply or questions
        if message.role == MessageRole.ASSISTANT and message.type == MessageType.TEXT:
            metrics = message.metadata.get("turn_metrics")
            foldable = self._foldable_ui_messages.pop(session_id, None)
            if foldable and foldable["name"]:
                # Finalize the foldable UI card
                body = {
                    "cardsV2": [
                        GoogleChatCardBuilder.build_foldable_ui_card(
                            session_id,
                            foldable["items"],
                            status="finished",
                            metrics=metrics,
                        )
                    ]
                }
                if message.channel_id and "threads" in message.channel_id:
                    body["thread"] = {"name": message.channel_id}
                try:
                    await asyncio.to_thread(
                        self._chat_service.spaces()
                        .messages()
                        .patch(
                            name=foldable["name"],
                            body=body,
                            updateMask="cardsV2",
                        )
                        .execute
                    )
                except Exception as e:
                    logger.error(f"Failed to final patch Google Chat foldable UI card: {e}", exc_info=True)

            # Now parse the final content and send it
            segments = parse_message_content(message.content)
            text_reply = ""
            choices: list[str] = []
            question_text = ""

            for seg in segments:
                if seg["type"] == "text":
                    text_reply += seg["content"]
                elif seg["type"] == "question":
                    question_text = seg["question"]
                    choices = seg["choices"]

            body = {}
            if message.channel_id and "threads" in message.channel_id:
                body["thread"] = {"name": message.channel_id}

            cards = []
            if text_reply.strip():
                cards.append(
                    {
                        "cardId": f"response_{session_id}_{int(time.time())}",
                        "card": {
                            "sections": [
                                {
                                    "widgets": [
                                        {
                                            "textParagraph": {
                                                "text": text_reply.strip(),
                                                "textSyntax": "MARKDOWN",
                                            }
                                        }
                                    ]
                                }
                            ]
                        },
                    }
                )

            if choices:
                cards.append(GoogleChatCardBuilder.build_question_card(session_id, question_text, choices))

            if cards:
                body["cardsV2"] = cards

            parent_space = message.channel_id.split("/threads/")[0] if message.channel_id else "spaces/unknown"
            try:
                await asyncio.to_thread(
                    self._chat_service.spaces()
                    .messages()
                    .create(
                        parent=parent_space,
                        body=body,
                        messageReplyOption="REPLY_MESSAGE_FALLBACK_TO_NEW_THREAD",
                    )
                    .execute
                )
                await self.gateway.db.update_message_status(message.id, MessageStatus.DELIVERED)
            except Exception as e:
                logger.error(f"Failed to send final message to Google Chat space: {e}", exc_info=True)

    def _build_gchat_custom_prompt(self, space_data: dict[str, Any], sender_name: str) -> str:
        """Build contextual system instructions injected into Google Chat sessions."""
        space_type = space_data.get("type", "SPACE")
        return (
            f"You are chatting with the user '{sender_name}' via a Google Chat {space_type} space.\n"
            "Format your replies cleanly using Google Chat supported markdown features:\n"
            "- Bold text using *asterisks* (e.g., *bold*).\n"
            "- Italic text using _underscores_ (e.g., _italic_).\n"
            "- Strikethrough text using ~tilde~ (e.g., ~strike~).\n"
            "- Monospace inline code using backticks (e.g., `code`).\n"
            "- Multi-line code blocks using triple backticks (e.g., ```py\\nprint('hi')\\n```).\n"
        )

    async def trigger_cronjob(
        self,
        channel_id: str,
        prompt_content: str,
        mention_user_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Trigger a scheduled cronjob in the specified Google Chat space/thread.

        Args:
            channel_id: Fully qualified space or thread resource name (e.g. 'spaces/AAAA/threads/BBBB').
            prompt_content: The prompt message content to run.
            mention_user_id: Optional user identifier (not natively highlighted here).
            **kwargs: Additional optional arguments.
        """
        custom_prompt = self._build_gchat_custom_prompt({"type": "SPACE"}, "System")

        msg_content = prompt_content
        if mention_user_id:
            # Google Chat user mention syntax: <users/USER_ID>
            msg_content = f"<users/{mention_user_id}> {msg_content}"

        tag = kwargs.get("tag")
        await self.trigger_cronjob_message(
            channel_id=channel_id,
            prompt_content=msg_content,
            sender_name="Cronjob",
            custom_prompt=custom_prompt,
            metadata={"is_cronjob": True},
            title=f"Google Chat Scheduled Job {channel_id.split('/')[-1]}",
            tag=tag,
        )
