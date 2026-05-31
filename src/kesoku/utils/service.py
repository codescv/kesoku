"""Service lifecycle and restart utilities for Kesoku."""

import asyncio
import logging
import os
import shutil
import sys
from collections.abc import Callable

logger = logging.getLogger(__name__)


def _get_kesoku_executable() -> str:
    """Resolve absolute path to the kesoku runner executable.

    Returns:
        Path string of the executable.
    """
    executable_dir = os.path.dirname(sys.executable)
    kesoku_path = os.path.join(executable_dir, "kesoku")
    if os.path.exists(kesoku_path):
        return kesoku_path
    return shutil.which("kesoku") or "kesoku"


async def restart_service(chatbot_id: str, stop_callback: Callable[[], None]) -> None:
    """Request and execute the service restart sequence.

    Args:
        chatbot_id: Chatbot requesting the restart.
        stop_callback: Action callback to stop the chatbot listener first.
    """
    logger.info(f"Chatbot '{chatbot_id}' requesting service restart.")
    stop_callback()

    kesoku_bin = _get_kesoku_executable()
    cmd = [kesoku_bin, "service", "restart"]

    service_user = os.environ.get("KESOKU_SERVICE_USER", "true") == "true"
    if service_user:
        cmd.append("--user")
    else:
        cmd.append("--system")

    instance_name = os.environ.get("KESOKU_SERVICE_INSTANCE_NAME")
    if instance_name:
        cmd.extend(["--name", instance_name])

    logger.info(f"Launching restart command: {' '.join(cmd)}")
    try:
        await asyncio.create_subprocess_exec(*cmd, start_new_session=True)
        logger.info("Successfully launched kesoku service restart command.")
    except Exception as e:
        logger.error(f"Failed to run restart command: {e}")
        # Fallback to in-place os.execv restart
        logger.info("Falling back to in-place os.execv restart...")
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception as fallback_error:
            logger.error(f"In-place fallback restart failed: {fallback_error}")
            raise e
