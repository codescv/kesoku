# 安全策略 (命令行执行限制)

由于 Kesoku 是一个自主智能体，拥有在宿主机上运行命令行指令（如文件读写、编译环境检查、运行测试脚本等）的能力，因此对命令执行的安全审计至关重要。Kesoku 提供了基于正则表达式的严格命令过滤机制。

---

## 🔒 核心配置项 (`[shell]`)

您可以在 `config.toml` 的 `[shell]` 分区中对安全过滤规则进行定义：

```toml
[shell]
enabled = true
use_shell = true
mode = "blocklist"
allowlist_patterns = ["^(echo|ls|pwd|cat|git|uv|grep|find|python|sed|awk)(\\s|$)"]
blocklist_patterns = ["(\\b|^)(rm|sudo|shutdown|reboot|mkfs|dd|chmod|chown)(\\b|\\s|$)"]
background_threshold_seconds = 300.0
```

### 参数详细说明：

1.  **`enabled`**（布尔值，默认：`true`）：
    是否启用终端命令行执行工具。如果设置为 `false`，Agent 将彻底失去调用系统命令行的能力。

2.  **`use_shell`**（布尔值，默认：`true`）：
    是否通过系统 Shell 执行命令（即 `subprocess.Popen(..., shell=True)`）。启用后支持管道符（`|`）、重定向（`>`）以及环境变量展开。

3.  **`mode`**（字符串，默认：`"blocklist"`）：
    *   `"blocklist"`（黑名单模式）：默认允许执行所有命令，除非命令匹配了 `blocklist_patterns` 中的正则表达式。
    *   `"allowlist"`（白名单模式）：默认禁止执行所有命令，除非命令完全匹配 `allowlist_patterns` 中的任一正则表达式。
4.  **`allowlist_patterns`** / **`blocklist_patterns`**（字符串列表）：
    用于审查完整命令字符串的正则表达式列表。

5.  **`background_threshold_seconds`**（浮点数，默认：`300.0`）：
    允许命令在前台运行的最长时间（秒）。如果一条命令执行时间超过此阈值，系统会自动将其转为后台非阻塞任务，防止会话卡死。

---

## ⚙️ 命令审查过滤流程

当 Agent 尝试运行一条 Shell 指令时：

1.  系统去除命令首尾的空白字符。
2.  **黑名单模式 (Blocklist Mode)**：
    *   逐个用 `blocklist_patterns` 里的正则去匹配该命令。
    *   一旦任一正则匹配成功（例如命令中包含 `sudo` 或试图运行 `rm` 删库），命令会**立即被拦截拒绝**，并向 Agent 返回安全警告。
3.  **白名单模式 (Allowlist Mode)**：
    *   逐个用 `allowlist_patterns` 里的正则去匹配该命令。
    *   如果**没有任何**正则能匹配该命令，命令会**立即被拦截拒绝**。
4.  如果命令通过安全检查，系统会在独立的子进程中执行该命令，并实时捕获其 `stdout` 与 `stderr` 流。
