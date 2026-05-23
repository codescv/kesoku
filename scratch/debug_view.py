import asyncio
from rich.console import Console
from kesoku.config import load_config
from kesoku.db import DatabaseManager
from kesoku.gateway.gateway import Gateway
from kesoku.agent.history import build_clean_history
from kesoku.cli_chat import _render_message

async def run_debug():
    load_config("/usr/local/google/home/chii/Developer/band/config.toml")
    gw = Gateway()
    console = Console()
    
    session_id = "6e2a4924"
    
    history = await build_clean_history(gw, session_id)
    print("\n--- RENDERED HISTORICAL MESSAGES ---")
    for m in history[-8:]:
        _render_message(console, m)

if __name__ == "__main__":
    asyncio.run(run_debug())
