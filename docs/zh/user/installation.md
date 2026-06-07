# 安装指南

本指南将指导您在本地机器或服务器上完成 Kesoku 的安装与配置。

---

## 📋 前提条件

在开始安装之前，请确保您的系统环境满足以下要求：

*   **Python**：版本需在 3.12 或更新。
*   **uv**：极速 Python 包管理与虚拟环境工具。
    *   如果您尚未安装 `uv`，可参考 [uv 安装说明](https://github.com/astral-sh/uv#installation)。通常，您可以通过以下命令快速安装：
        ```bash
        curl -LsSf https://astral-sh.uv.run/install.sh | sh
        ```
*   **Git**：用于克隆项目仓库。

---

## 📦 开始安装

按照以下步骤克隆并运行 Kesoku：

### 步骤 1：克隆仓库
将代码仓库克隆到您的本地目录：

```bash
git clone https://github.com/codescv/kesoku.git
cd kesoku
```

### 步骤 2：同步依赖包
Kesoku 使用 `uv` 管理虚拟环境与锁定的依赖项。要自动创建本地虚拟环境并安装所有运行所需的依赖，只需执行：

```bash
uv sync
```
此命令将在项目根目录创建 `.venv/` 虚拟环境，并同步 `uv.lock` 中指定的所有包。

### 步骤 3：验证命令行工具
通过 `uv run` 执行以下命令，以确认命令行入口是否正常工作：

```bash
uv run kesoku --help
```
如果安装成功，终端将输出 `kesoku` 命令行工具的帮助信息（展示 `chat`、`start`、`service`、`init`、`wechat` 等子命令）。

---

## 🛠️ 开发者安装（可选）

如果您计划对 Kesoku 贡献代码、运行单元测试或本地预览此文档网站，请同步开发组依赖项：

```bash
# 同步包含 ruff, pytest, 和 mkdocs 在内的所有开发依赖
uv sync --all-groups
```
