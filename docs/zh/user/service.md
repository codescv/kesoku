# 系统服务管理 (systemd / launchd)

为了支持将 Kesoku 部署为长期运行的后台守护进程（例如 Discord 或 Google Chat 聊天机器人），命令行工具内置了 `kesoku service` 服务管理命令组。该工具会自动检测宿主机操作系统类型，在 Linux 上自动生成并管理 `systemd` 服务单元，在 macOS 上自动生成 `launchd` 属性列表配置（`.plist`）。

---

## 📥 安装后台服务

将 Kesoku 注册为开机自启的系统后台服务：

### 1. 用户级服务 (User-Level, 推荐)
运行在当前普通用户权限范围内，**不需要** root / sudo 管理员权限：
```bash
uv run kesoku service install -c private/config.toml
```

### 2. 系统级服务 (System-Level)
运行在全局系统范围内，需要 `sudo` 权限来安装：
```bash
sudo uv run kesoku service install --system -c private/config.toml
```

### 🔧 注入与继承环境变量
默认情况下，服务安装程序会自动从您执行安装命令的终端 shell 中继承以下环境变量（如果存在）：
`PATH`, `HTTP_PROXY`, `HTTPS_PROXY`, `GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `GOOGLE_GENAI_USE_VERTEXAI`, `DISCORD_TOKEN`。

如果您需要显式覆盖或为服务注入其他的环境变量，可以使用 `-e` / `--env` 选项（可以多次声明）：
```bash
uv run kesoku service install -c config.toml -e GEMINI_API_KEY=your_api_key -e CUSTOM_PORT=8080
```

### 🔍 预览配置 (Dry-Run)
在不真正写入系统目录的情况下，在终端打印出生成的 systemd unit 文件或 plist 文件配置内容：
```bash
uv run kesoku service install --dry-run -c config.toml
```

---

## 🕹️ 控制服务运行

服务安装成功后，您可以使用以下子命令控制守护进程的运行状态：

### 启动服务
```bash
uv run kesoku service start
```

### 停止服务
```bash
uv run kesoku service stop
```

### 重启服务
```bash
uv run kesoku service restart
```

### 查询运行状态
显示当前服务是否处于 active 运行中，或是已经退出，并展示其 PID 信息：
```bash
uv run kesoku service status
```

*注意：如果您的服务是通过 `--system` 安装的，控制命令也必须带上 `--system` 标识（如 `uv run kesoku service start --system`）。*

---

## 📜 实时查看日志

后台守护进程的所有标准输出 (`stdout`) 和错误输出 (`stderr`) 都会被自动重定向到操作系统标准的日志管理守护进程中（Linux 上为 `journald`，macOS 上为系统日志）。

您可以通过命令行直接查看或跟踪日志流：

### 查看最近日志
默认打印最后 50 行日志：
```bash
uv run kesoku service logs
```

### 实时流式跟踪日志
持续输出最新的日志信息（效果等同于 `tail -f`）：
```bash
uv run kesoku service logs -f
```

### 限制日志输出行数
查看指定行数的历史日志：
```bash
uv run kesoku service logs -n 100
```

---

## 🗑️ 卸载后台服务
停止运行服务，取消开机自启，并干净地清除系统中残留的所有 Kesoku 服务配置文件：
```bash
uv run kesoku service uninstall
```
