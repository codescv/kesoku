# Safety Settings (Shell Execution)

Since Kesoku is an autonomous agent capable of executing terminal commands (such as file modifications, environment checks, and script runs), securing command execution is critical. Kesoku provides robust regex-based filtering of terminal commands before execution.

---

## 🔒 Configuration (`[shell]`)

Command filtering is configured under the `[shell]` section in `config.toml`:

```toml
[shell]
enabled = true
use_shell = true
mode = "blocklist"
allowlist_patterns = ["^(echo|ls|pwd|cat|git|uv|grep|find|python|sed|awk)(\\s|$)"]
blocklist_patterns = ["(\\b|^)(rm|sudo|shutdown|reboot|mkfs|dd|chmod|chown)(\\b|\\s|$)"]
background_threshold_seconds = 300.0
```

### Key Parameters:

1.  **`enabled`** (boolean, default: `true`):
    Enables or disables the command execution tool entirely. If set to `false`, the agent will not be able to execute any command line inputs.
2.  **`use_shell`** (boolean, default: `true`):
    If `true`, command execution uses `subprocess.Popen(..., shell=True)`. This allows operators like pipes (`|`), redirection (`>`), and environment expansions.
3.  **`mode`** (string, default: `"blocklist"`):
    *   `"blocklist"`: Commands are allowed by default, unless they match one of the regex patterns in `blocklist_patterns`.
    *   `"allowlist"`: Commands are blocked by default, unless they match one of the regex patterns in `allowlist_patterns`.
4.  **`allowlist_patterns`** / **`blocklist_patterns`** (list of strings):
    List of regular expressions used to inspect the full command string.
5.  **`background_threshold_seconds`** (float, default: `300.0`):
    The maximum execution duration allowed for a command in the foreground. If a command runs longer than this threshold, it is automatically safely detached into a background execution job.

---

## ⚙️ How Command Inspection Works

When the agent attempts to run a terminal command:
1.  The command string is stripped of leading/trailing whitespaces.
2.  In **Blocklist Mode**:
    *   The command is evaluated against each pattern in `blocklist_patterns`.
    *   If any regex matches (e.g. command contains `sudo` or `rm -rf`), execution is rejected immediately, and a warning is returned to the agent.
3.  In **Allowlist Mode**:
    *   The command is evaluated against each pattern in `allowlist_patterns`.
    *   If **none** of the regexes match, execution is rejected immediately.
4.  If approved, the command runs within a subprocess container, and its `stdout` and `stderr` streams are captured.
