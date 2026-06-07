# Service Management (systemd / launchd)

To run Kesoku as a continuous background daemon (for Discord or Google Chat bots), the CLI provides a built-in service manager command group: `kesoku service`. This manager automatically detects your OS and generates/manages a `systemd` unit on Linux or a `launchd` property list (`.plist`) on macOS.

---

## 📥 Installing the Service

To register Kesoku as a startup background service:

### 1. User-Level Service (Recommended)
This runs under your user account scope and does not require administrative root/sudo permissions:
```bash
uv run kesoku service install -c private/config.toml
```

### 2. System-Level Service (Global)
This runs globally on the OS and requires administrative `sudo` permissions to install:
```bash
sudo uv run kesoku service install --system -c private/config.toml
```

### 🔧 Injecting Environment Variables
By default, the service installer automatically inherits these environment variables from your current terminal environment if they are set:
`PATH`, `HTTP_PROXY`, `HTTPS_PROXY`, `GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `GOOGLE_GENAI_USE_VERTEXAI`, `DISCORD_TOKEN`.

To manually append or override environment variables for the service daemon, use the `-e` / `--env` options (which can be declared multiple times):
```bash
uv run kesoku service install -c config.toml -e GEMINI_API_KEY=your_api_key -e CUSTOM_PORT=8080
```

### 🔍 Dry-Run Preview
To preview the generated config file contents (e.g. the systemd `.service` file or plist structure) without writing to the disk:
```bash
uv run kesoku service install --dry-run -c config.toml
```

---

## 🕹️ Controlling the Service

Once installed, use the following subcommand wrappers to control the daemon process:

### Start the Service
```bash
uv run kesoku service start
```

### Stop the Service
```bash
uv run kesoku service stop
```

### Restart the Service
```bash
uv run kesoku service restart
```

### Query Active Status
Shows whether the daemon is active, running, or failed, along with its process ID:
```bash
uv run kesoku service status
```

*Note: If you installed the service with the `--system` flag, you must append the `--system` option to these control commands too (e.g. `uv run kesoku service start --system`).*

---

## 📜 Reviewing Service Logs

All `stdout` and `stderr` logs from the background daemon are automatically routed to standard OS logging daemons (`journald` on Linux and system logs on macOS).

You can follow or inspect logs directly through the CLI:

### Show Recent Logs
Display the last 50 log lines:
```bash
uv run kesoku service logs
```

### Stream Live Logs
Follow logs in real-time (similar to `tail -f`):
```bash
uv run kesoku service logs -f
```

### Limit Log Lines
Specify how many lines of history to display:
```bash
uv run kesoku service logs -n 100
```

---

## 🗑️ Uninstalling the Service
To stop the service, disable boot-time autostart, and remove all service configuration files cleanly from your system:
```bash
uv run kesoku service uninstall
```
