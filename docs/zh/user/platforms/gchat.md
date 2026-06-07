# Google Chat 平台接入指南 (GCP 配置)

本指南详述了为集成 Kesoku AI Agent 所需进行的 Google Cloud Platform (GCP) 项目配置、Google Cloud Pub/Sub 订阅创建、服务账号（Service Account）设置以及 Google Chat API 的连接设置。

---

## 1. 启用 Google Cloud APIs

首先，在您的 Google Cloud 控制台中启用所需的 API：

1. 打开 [Google Cloud Console](https://console.cloud.google.com/)。
2. 选择或新建一个 GCP 项目。
3. 导航至 **API 和服务 > 库** (APIs & Services > Library)。
4. 搜索并启用以下两个 API：
   - **Google Chat API**
   - **Cloud Pub/Sub API**

---

## 2. 配置 Cloud Pub/Sub 消息队列

Google Chat 将用户交互事件发布到 Pub/Sub 主题（Topic）中，Kesoku 则通过挂载在该主题下的拉取订阅（Pull Subscription）来接收事件。

### 步骤 A：创建 Pub/Sub 主题 (Topic)
1. 在控制台中，打开 **Pub/Sub > 主题** (Topics) 页面。
2. 点击 **创建主题** (Create Topic)。
3. 将 **主题 ID** 设置为 `kesoku-chat-events`（或自定义的其他描述性名称）。
4. 取消勾选“添加默认订阅”选项（我们稍后将创建自定义订阅）。
5. 点击 **创建**。

### 步骤 B：向 Google Chat 授权发布权限
Google Chat 服务需要获取向您刚刚创建的 Pub/Sub 主题发布消息的权限。

1. 在 `kesoku-chat-events` 主题详情页面中，点击右上角的 **显示信息面板** (Show Info Panel)。
2. 在 **权限** 选项卡下，点击 **添加成员** (Add Principal)。
3. 在 **新的成员** 输入框中，填入 Google Chat API 的官方系统服务账号：
   ```
   chat-api-push@system.gserviceaccount.com
   ```
4. 为其分配角色：**Pub/Sub 发布者** (`roles/pubsub.publisher`)。
5. 点击 **保存**。

### 步骤 C：创建拉取订阅 (Pull Subscription)
1. 在该主题页面上，点击 **创建订阅** (Create Subscription)。
2. 将 **订阅 ID** 设置为 `kesoku-chat-sub`。
3. 确保 **递送类型** 勾选为 **拉取 (Pull)**。
4. 保持其他默认选项不变，点击底部 **创建**。

---

## 3. 创建与授权服务账号 (Service Account)

Kesoku 使用服务账号进行 GCP 接口的权限验证（包括拉取 Pub/Sub 消息及调用 Chat 接口发送回复）。

### 步骤 A：创建服务账号
1. 导航至 **IAM & 管理 > 服务账号** (IAM & Admin > Service Accounts)。
2. 点击 **创建服务账号**。
3. 设置名称为 `kesoku-chat-agent`，点击 **创建并继续**。
4. 在角色分配步骤中，为该账号赋予以下角色权限：
   - **Pub/Sub 订阅者** (`roles/pubsub.subscriber`)：允许拉取并确认订阅的消息。
   - **Pub/Sub 查看者** (`roles/pubsub.viewer`)：用于验证连接和检索主题/订阅元数据。
5. 点击 **继续** 和 **完成**。

### 步骤 B：免密钥配对 (Service Account 凭证模拟方案)
在一些严苛的企业级云环境中（例如 Google 内部），出于安全政策考虑，通常禁止创建或下载本地静态私钥 JSON 文件。此时，您可以使用 **服务账号模拟 (Service Account Impersonation)** 方案。

该方案将授权您的个人企业云账号临时模拟该服务账号来运行 Agent。

#### 1. 启用凭证 API
确保您的 GCP 项目启用了 **IAM Service Account Credentials API**：

- 在 **API 和服务 > 库** 中搜索并启用 `iamcredentials.googleapis.com`。

#### 2. 向您的企业账号授权模拟权限
向您的个人企业邮箱授予该服务账号的 **Service Account Token Creator** 角色权限：

1. 在 **IAM & 管理 > 服务账号** 页面中，点击刚刚创建的 `kesoku-chat-agent` 服务账号。
2. 切换到 **权限** (Permissions) 选项卡。
3. 点击 **授予访问权限** (Grant Access)。
4. 在成员中输入您的个人企业邮箱。
5. 分配角色：**Service Account Token Creator**。
6. 点击 **保存**。

#### 3. 本地终端完成认证
在本地开发环境或服务器上运行以下命令登录，它将自动配置本地 GCP SDK，临时代表服务账号执行身份验证：

```bash
gcloud auth application-default login --impersonate-service-account=kesoku-chat-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com
```
启动 Kesoku 时，Google 官方的认证库和 Pub/Sub 客户端库会自动读取此默认凭证并代入，无需本地配置静态 JSON 密钥文件。

#### 4. 在配置文件中声明模拟 (替代方案)
作为替代方案，您也可以只在终端运行标准的默认身份登录 (`gcloud auth application-default login`)，然后在 `config.toml` 文件的 `[google_chat]` 配置块中，显式填入 `impersonate_service_account`（填入您的服务账号邮箱地址）进行凭证中转。

---

## 4. 关联 Google Chat API 配置

在 GCP 控制台中将 Pub/Sub 主题与您的 Google Chat 机器人应用进行关联绑定：

1. 打开 GCP Console，进入 **Google Chat API > 配置** (Configuration) 页面。
2. 完善应用的基础信息：
   - **应用名称 (App name)**：Kesoku AI Agent
   - **头像 (Avatar URL)**：(可选) 机器人的头像链接
   - **描述 (Description)**：自主 AI 编程智能体 (Autonomous AI Coding Agent)
3. 向下滚动至 **交互功能 (Interactive features)** 区域，勾选启用。
4. 在 **连接设置 (Connection settings)** 处，勾选 **Cloud Pub/Sub**。
5. 在 **Pub/Sub 主题** 输入框中，填入您刚刚创建的主题的完整资源路径：
   ```
   projects/YOUR_PROJECT_ID/topics/kesoku-chat-events
   ```
6. 在 **可见性 (Visibility)** 处，配置谁可以安装此聊天机器人（例如，仅允许您所在企业组织下的用户安装）。
7. 点击 **保存**。

---

## 5. 常见交互卡片认证 FAQ

### 问：发送或更新交互式卡片（Chat Cards）需要特殊的 OAuth 授权或用户登录吗？

**不需要。**

当用户点击卡片上的交互按钮时，Google Chat 会直接将 `CARD_CLICKED` 交互事件发布到 Pub/Sub 主题。Kesoku 会在后台异步拉取该事件并做出处理。

### 权限与认证范围 (Scopes)：
- **App 认证 (App Authentication)**：由于 Kesoku 通过 Pub/Sub 隐藏在防火墙后异步工作，所有消息发送与卡片更新均由服务账号调用 Chat API 来触发。
- 服务账号使用标准的 **Chat Bot** 授权范围（Scope）：
  ```
  https://www.googleapis.com/auth/chat.bot
  ```
  该权限范围授权机器人在已被邀请加入的群组或空间中以自己（App）的名义发送消息。

- **免登录授权**：与 Sheets 或 Drive 等需要用户弹出 OAuth 授权同意书并生成用户级 Token 的应用不同，Google Chat App 在被用户邀请/加入到空间的那一刻起，就已经获取了合法的 `chat.bot` 权限。整个过程中不需要任何终端用户的二次登录确认。

### 问：是否需要赋予服务账号特殊的 IAM 读写角色权限来发送消息？

**不需要。**

1. **服务账号侧**：在 Google Chat 生态中，聊天应用只要被管理员/用户邀请进了对应的空间 (Space) 或线程，就已经自动继承了读写消息的完整权限。您无需在 GCP IAM 中为其再分配类似于“Google Chat 写入者”等任何额外的特殊云端写角色。
2. **本地模拟用户侧**：您的个人企业账号本身不需要任何 Google Chat API 权限。当您执行服务账号模拟登录时，GCP 会验证您的 `Service Account Token Creator` 权限并颁发临时 Token。对于 Chat 接口而言，调用者身份依旧是服务账号（即机器人应用本身），它完全有权在其所在的会话空间中收发消息。
