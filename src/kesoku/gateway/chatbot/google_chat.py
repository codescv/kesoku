"""Google Chat chatbot adapter for Kesoku AI Agent framework.

Connects Google Chat spaces with the internal Kesoku Gateway using
a Google Cloud Pub/Sub Pull Subscription and GCP public APIs.
"""

import asyncio
import json
import time
from typing import Any

import google.auth
from google.auth import impersonated_credentials
from google.cloud import pubsub_v1
from googleapiclient.discovery import Resource, build

from kesoku.config import get_config
from kesoku.constants import (
    ROLE_ASSISTANT,
    ROLE_USER,
    STATUS_DELIVERED,
    STATUS_PENDING_AGENT,
    TYPE_TEXT,
)
from kesoku.db import Message
from kesoku.gateway.chatbot.base import Chatbot, parse_message_content
from kesoku.gateway.gateway import Gateway
from kesoku.logger import setup_logger

logger = setup_logger(__name__)


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
            raise ValueError(
                "Google Chat is enabled but project_id, topic_id, or subscription_id are not configured."
            )

        self._running = False
        self._pubsub_task: asyncio.Task[None] | None = None

        # Load credentials and initialize Google APIs
        self._credentials, self._project_id = self._load_credentials()
        self._subscriber_client = pubsub_v1.SubscriberClient(credentials=self._credentials)
        self._subscription_path = self._subscriber_client.subscription_path(
            self.config.project_id, self.config.subscription_id
        )

        # Build standard Google Chat API client
        self._chat_service: Resource = build("chat", "v1", credentials=self._credentials)

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
        streaming_pull_future = self._subscriber_client.subscribe(
            self._subscription_path, callback=_pubsub_callback
        )

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

        # 1. Direct / Space Messages
        if event_type == "MESSAGE":
            await self._handle_incoming_message(event)

        # 2. Interactive Card Button Click
        elif event_type == "CARD_CLICKED":
            await self._handle_card_interaction(event)

        # 3. Added to Space
        elif event_type == "ADDED_TO_SPACE":
            await self._handle_added_to_space(event)

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
        text = message_data.get("text", "").strip()
        space_name = space_data.get("name")  # e.g., "spaces/AAAAxxxx"
        thread_name = thread_data.get("name")  # e.g., "spaces/AAAAxxxx/threads/YYYY"

        # Security: Filter via user allowlist if configured
        if self.config.user_allowlist:
            sender_email = sender_data.get("email")
            if sender_id not in self.config.user_allowlist and sender_email not in self.config.user_allowlist:
                logger.warning(f"Google Chat: Ignoring unauthorized sender {sender_name} ({sender_id}/{sender_email})")
                return

        if not text:
            return

        # Define the thread context as the logical channel ID
        channel_id = thread_name if thread_name else space_name

        # Resolve or create Kesoku session mapped to the thread/space context
        session = await self.gateway.get_session_by_channel(self.chatbot_id, channel_id)
        if not session:
            title = f"Google Chat Session: {text[:30]}"
            session = await self.gateway.create_session(
                session_id=None,
                title=title,
                custom_prompt=self._build_gchat_custom_prompt(space_data, sender_name),
            )
            # Save session metadata
            await self.gateway.update_session_updated_at(session.id)

        # Construct Gateway Message
        user_msg = Message(
            session_id=session.id,
            chatbot_id=self.chatbot_id,
            channel_id=channel_id,
            sender=sender_name,
            role=ROLE_USER,
            type=TYPE_TEXT,
            content=text,
            status=STATUS_PENDING_AGENT,
            timestamp=time.time(),
        )

        # Post user message to trigger agent processing loop
        await self.gateway.post(user_msg)

    async def _handle_card_interaction(self, event: dict[str, Any]) -> None:
        """Route action clicks on Google Chat interactive cards.

        Args:
            event: The decoded JSON interaction event payload.
        """
        action_data = event.get("action", {})
        function_name = action_data.get("actionMethodName")
        parameters = {p["key"]: p["value"] for p in action_data.get("parameters", [])}

        session_id = parameters.get("session_id")
        if not session_id:
            return

        logger.info(f"Google Chat button clicked: {function_name} for session: {session_id}")

        if function_name == "stop_turn":
            # Trigger worker cancellation via gateway registered agent dispatcher
            if self.gateway.agent:
                await self.gateway.agent.stop_session_worker(session_id)
                logger.info(f"Google Chat requested stop turn for session: {session_id}")

        elif function_name == "clear_session":
            # Recursively delete session database records and workspace staging directories
            await self.gateway.delete_session(session_id)
            logger.info(f"Google Chat requested clear session for: {session_id}")

        elif function_name == "submit_choice":
            choice_val = parameters.get("choice")
            space_data = event.get("space", {})
            thread_data = event.get("message", {}).get("thread", {})
            thread_name = thread_data.get("name")
            space_name = space_data.get("name")
            channel_id = thread_name if thread_name else space_name

            if choice_val:
                # Post choice to the gateway as a new user message
                choice_msg = Message(
                    session_id=session_id,
                    chatbot_id=self.chatbot_id,
                    channel_id=channel_id,
                    sender=event.get("user", {}).get("displayName", "User"),
                    role=ROLE_USER,
                    type=TYPE_TEXT,
                    content=choice_val,
                    status=STATUS_PENDING_AGENT,
                    timestamp=time.time(),
                )
                await self.gateway.post(choice_msg)

    async def _handle_added_to_space(self, event: dict[str, Any]) -> None:
        """Greet the user when the bot is added to a space.

        Args:
            event: The decoded JSON interaction event payload.
        """
        space_data = event.get("space", {})
        space_name = space_data.get("name")

        logger.info(f"Google Chat: Bot was added to space {space_name}")

        welcome_body = {
            "text": (
                "👋 Hello! I am the Kesoku Autonomous AI Coding Agent. "
                "Ask me questions or mention me to begin coding!"
            )
        }
        await asyncio.to_thread(
            self._chat_service.spaces().messages().create(parent=space_name, body=welcome_body).execute
        )

    async def handle_message(self, message: Message) -> None:
        """Process and send outgoing assistant messages to the Google Chat API.

        Args:
            message: The Message instance to handle.
        """
        # Only handle messages destined for this chatbot adapter
        if message.chatbot_id != self.chatbot_id:
            return

        logger.debug(f"Google Chat processing outgoing message ID {message.id}")

        # Check if the message contains markdown file patterns or questions
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

        # Prepare Google Chat payload
        body: dict[str, Any] = {}
        if text_reply.strip():
            body["text"] = text_reply.strip()

        # Append thread key if replying to a thread channel context
        if message.channel_id and "threads" in message.channel_id:
            body["thread"] = {"name": message.channel_id}

        # Add Interactive Card UI controls for final text messages
        if message.role == ROLE_ASSISTANT and message.type == TYPE_TEXT:
            if choices:
                # Format multiple-choice questions as a beautiful interactive card
                body["cardsV2"] = [self._build_question_card(message.session_id, question_text, choices)]
            else:
                # Attach Stop / Clear / Trajectory controls card
                body["cardsV2"] = [self._build_control_card(message.session_id)]

        # Send asynchronously inside execution thread pool
        parent_space = message.channel_id.split("/threads/")[0] if message.channel_id else "spaces/unknown"
        try:
            await asyncio.to_thread(
                self._chat_service.spaces()
                .messages()
                .create(parent=parent_space, body=body)
                .execute
            )
            # Update status to delivered
            await self.gateway.update_message_status(message.id, STATUS_DELIVERED)
        except Exception as e:
            logger.error(f"Failed to send outgoing message to Google Chat space: {e}", exc_info=True)

    def _build_control_card(self, session_id: str) -> dict[str, Any]:
        """Construct the interactive control panel card.

        Args:
            session_id: Active session ID.

        Returns:
            A cardsV2 dictionary structure.
        """
        return {
            "cardId": f"control_{session_id}",
            "card": {
                "header": {
                    "title": "Kesoku Agent Control Panel",
                    "subtitle": "Manage active turn and session",
                },
                "sections": [
                    {
                        "widgets": [
                            {
                                "buttonList": {
                                    "buttons": [
                                        {
                                            "text": "Stop Turn 🛑",
                                            "onClick": {
                                                "action": {
                                                    "function": "stop_turn",
                                                    "parameters": [{"key": "session_id", "value": session_id}],
                                                }
                                            },
                                        },
                                        {
                                            "text": "Clear Session ♻️",
                                            "onClick": {
                                                "action": {
                                                    "function": "clear_session",
                                                    "parameters": [{"key": "session_id", "value": session_id}],
                                                }
                                            },
                                        },
                                    ]
                                }
                            }
                        ]
                    }
                ],
            },
        }

    def _build_question_card(self, session_id: str, question: str, choices: list[str]) -> dict[str, Any]:
        """Construct the interactive multiple-choice question card.

        Args:
            session_id: Active session ID.
            question: The question prompt text.
            choices: A list of buttons values.

        Returns:
            A cardsV2 dictionary structure.
        """
        buttons = []
        for choice in choices:
            buttons.append(
                {
                    "text": choice,
                    "onClick": {
                        "action": {
                            "function": "submit_choice",
                            "parameters": [
                                {"key": "session_id", "value": session_id},
                                {"key": "choice", "value": choice},
                            ],
                        }
                    },
                }
            )

        return {
            "cardId": f"question_{session_id}",
            "card": {
                "header": {"title": "Interactive Choice Question"},
                "sections": [
                    {"widgets": [{"textParagraph": {"text": question}}]},
                    {"widgets": [{"buttonList": {"buttons": buttons}}]},
                ],
            },
        }

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
