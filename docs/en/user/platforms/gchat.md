# Google Chat GCP Setup Guide

This guide provides the step-by-step instructions for configuring a Google Cloud Platform (GCP) project, Google Cloud Pub/Sub, service accounts, and Google Chat API Connection settings to integrate the Kesoku AI Agent.

---

## 1. Enable Google Cloud APIs

First, enable the required APIs in your Google Cloud Console:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Select or create a Google Cloud project.
3. Navigate to **APIs & Services > Library**.
4. Search for and enable the following APIs:
   - **Google Chat API**
   - **Cloud Pub/Sub API**

---

## 2. Configure Cloud Pub/Sub

Google Chat publishes interaction events to a Pub/Sub topic. Kesoku pulls events from a subscription attached to that topic.

### Step A: Create a Pub/Sub Topic
1. Go to the **Pub/Sub > Topics** page in the Cloud Console.
2. Click **Create Topic**.
3. Set **Topic ID** to `kesoku-chat-events` (or another descriptive name).
4. Leave "Add a default subscription" unchecked (we will create a custom one later).
5. Click **Create**.

### Step B: Grant Publisher Permission to Google Chat
Google Chat needs permission to publish events to your newly created topic.

1. On your `kesoku-chat-events` topic page, click **Show Info Panel** (top right).
2. Under the **Permissions** tab, click **Add Principal**.
3. In **New principals**, enter the official Google Chat API system service account:
   ```
   chat-api-push@system.gserviceaccount.com
   ```
4. Assign the role **Pub/Sub Publisher** (`roles/pubsub.publisher`).
5. Click **Save**.

### Step C: Create a Pull Subscription
1. On the topic page, click **Create Subscription**.
2. Set **Subscription ID** to `kesoku-chat-sub`.
3. Ensure the **Delivery Type** is set to **Pull**.
4. Under **Message retention**, keep the default settings.
5. Click **Create**.

---

## 3. Create and Authorize the Service Account

Kesoku uses a Service Account to authenticate GCP requests (pulling messages and posting replies).

### Step A: Create the Service Account
1. Go to **IAM & Admin > Service Accounts**.
2. Click **Create Service Account**.
3. Set the name to `kesoku-chat-agent` and click **Create and Continue**.
4. Assign the following roles to the Service Account:
   - **Pub/Sub Subscriber** (`roles/pubsub.subscriber`): Allows pulling messages from the subscription.
   - **Pub/Sub Viewer** (`roles/pubsub.viewer`): Required for checking metadata and connection statuses.
5. Click **Continue** and **Done**.

### Step B: Key-Less Setup (Impersonation Workaround)
In secure or corporate environments (like Google) where creating or downloading static service account JSON keys is prohibited, you can use **Service Account Impersonation**.

To do this, configure your local user identity (which runs Kesoku) to impersonate the service account.

#### 1. Enable IAM Credentials API
Ensure the **IAM Service Account Credentials API** is enabled in your GCP project:

- Navigate to **APIs & Services > Library** and enable `iamcredentials.googleapis.com`.

#### 2. Grant Impersonation Permission to your Corporate User
Grant your personal corporate email address the **Service Account Token Creator** role (`roles/iam.serviceAccountTokenCreator`) on the service account:

1. In **IAM & Admin > Service Accounts**, click on `kesoku-chat-agent`.
2. Click the **Permissions** tab.
3. Click **Grant Access**.
4. Add your corporate email address as a principal.
5. Select the role **Service Account Token Creator**.
6. Click **Save**.

#### 3. Authenticate Local Environment via CLI (Option A - Zero Code Changes)
Run the following command on your local development machine to log in and automatically configure all local GCP SDK libraries to impersonate the bot service account:

```bash
gcloud auth application-default login --impersonate-service-account=kesoku-chat-agent@YOUR_PROJECT_ID.iam.gserviceaccount.com
```
When Kesoku starts up, the Google Pub/Sub and Auth libraries will automatically read these credentials and impersonate the service account seamlessly without requiring a JSON key file.

#### 4. Authenticate via Configuration (Option B - Explicit Config)
Alternatively, you can leave your local ADC logged in normally (`gcloud auth application-default login`) and explicitly specify the target impersonated service account email in `config.toml` (see the Implementation Plan).

---

## 4. Configure Google Chat API Connection Settings

Link your GCP configuration to the Google Chat App:

1. Go to **Google Chat API > Configuration** in the GCP Console.
2. Fill out the App Info details:
   - **App name**: Kesoku AI Agent
   - **Avatar URL**: (Optional) Link to a custom avatar
   - **Description**: Autonomous AI Coding Agent
3. Scroll down to the **Interactive features** section and enable it.
4. Under **Connection settings**, select **Cloud Pub/Sub**.
5. In **Pub/Sub topic**, enter the fully qualified resource name of your topic:
   ```
   projects/YOUR_PROJECT_ID/topics/kesoku-chat-events
   ```
6. Under **Visibility**, configure who can install the app (e.g., private to your domain or specific email addresses).
7. Click **Save**.

---

## 5. Authentication Scopes & Special Setup for Chat Cards

### Q: Is there a special setup or authentication required for Google Chat Cards?

**No special setup or user-level authentication is required for Chat Cards.** 

When a user clicks a button on a Card, Google Chat sends a `CARD_CLICKED` event to the Pub/Sub topic. Kesoku handles these events entirely asynchronously. 

### Scopes & Authentication:
- **App Authentication**: Because Kesoku runs asynchronously behind a firewall using Pub/Sub, all replies (including card renders, updates, or message posts) are initiated by the Service Account calling the Chat API.
- The Service Account uses the standard **Chat Bot** authorization scope:
  ```
  https://www.googleapis.com/auth/chat.bot
  ```
  This scope authorizes the bot to act as itself inside any space it has been added to.

- **Zero User Login Screen**: Unlike other Google Workspace integrations (like Sheets or Drive) that require user OAuth consent screens and individual authorization tokens, Google Chat apps using App Authentication (`chat.bot` scope) authorize instantly as soon as the bot is invited/added to the space by a user. No user authorization flow is needed!

### Q: Do I need to grant the Service Account or the local user special IAM permissions to send messages?

**No.** 

1. **For the Service Account**: In the Google Chat ecosystem, a chatbot app automatically inherits permissions to read/write messages inside any Google Chat Space or Thread it has been invited or added to. No custom or extra GCP IAM roles (such as a "Google Chat Writer" role) need to be assigned to the Service Account within the GCP console for message delivery.
2. **For the local user (Impersonation context)**: Your personal corporate user account does not need any Google Chat API permissions. When you run `gcloud auth application-default login --impersonate-service-account=...`, the Google Cloud IAM token service verifies your `Service Account Token Creator` permission, logs you in, and generates temporary security tokens representing the Service Account itself. To the Chat API, the caller is the Service Account (the App/Bot), which has permissions to interact naturally inside its spaces.

