# 安装指南

本指南将指导您在本地机器或服务器上完成 Kesoku 的安装与配置。

---

## 📋 前提条件

在开始安装之前，请确保您的系统环境满足以下要求：

*   **Python**：版本需在 3.12 或更新。
*   **uv**：极速 Python 包与虚拟环境管理工具。
    *   如果您尚未安装 `uv`，运行以下命令快速安装：
        ```bash
        curl -LsSf https://astral-sh.uv.run/install.sh | sh
        ```
*   **Git**：从仓库直接安装时需要使用。

---

## 📦 安装 Kesoku (推荐方式)

最推荐的安装方式是使用 `uv tool install` 将 Kesoku 作为全局命令行工具安装。`uv` 会自动为您维护虚拟环境隔离，并将 `kesoku` 可执行文件添加到系统的环境变量中。

### 选项 A：直接从 GitHub 安装
运行以下命令从主分支安装最新版本的 Kesoku：

```bash
uv tool install git+https://github.com/codescv/kesoku.git
```

### 选项 B：从本地克隆代码安装
如果您已经将项目代码克隆到了本地：

```bash
git clone https://github.com/codescv/kesoku.git
cd kesoku
uv tool install .
```

### 验证安装
在终端输入以下命令验证 `kesoku` 可执行程序是否正常工作：

```bash
kesoku --help
```
如果安装成功，终端将输出命令行帮助信息，列出可用的子命令（如 `chat`、`start`、`service`、`init`、`wechat`、`memory` 等）。

---

## 🛠️ 开发者安装

如果您计划参与 Kesoku 的开发、运行单元测试或修改代码：

1. 克隆代码库，并将其安装为**可编辑的全局工具**（Editable Tool），以便您对本地代码的任何修改能实时反映到 `kesoku` 指令中：
   ```bash
   git clone https://github.com/codescv/kesoku.git
   cd kesoku
   uv tool install -e .
   ```
2. 在项目根目录下创建本地虚拟环境并安装所有开发组相关的依赖（用于运行单元测试、格式化代码或本地预览文档）：
   ```bash
   uv sync --all-groups
   ```
3. 运行测试请使用 `uv run pytest`，编译/构建文档请使用 `uv run mkdocs build`。
