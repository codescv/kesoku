# Integration test: Shell Command Execution

Use configuration file: `private/config.toml`

# Steps
- Re-Initialize the db
- Use `kesoku chat` to request the agent to execute a shell command (e.g., `echo Hello Kesoku Shell`)
- Verify that the agent uses the `run_shell_command` tool and successfully returns the output
- Check that the session staging directory was created inside `private/sessions/`
