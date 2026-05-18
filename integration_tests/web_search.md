# Integration test: Web Search Grounding

Use configuration file: `private/config.toml`

# Steps
- Re-Initialize the db and workspace using `kesoku init -w private --overwrite-db`
- Use `kesoku chat` to ask a question requiring live web search grounding (e.g., "Who won the Super Bowl in 2025?")
- Verify that the agent invokes the `web_search` tool and successfully returns the grounded answer along with formatted source URLs.
