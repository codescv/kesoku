# Role Playing & Persona Configuration

Kesoku features a flexible and dynamic role-playing system, allowing users or the agent itself to switch characters (personas) instantly during a chat session.

---

## 🎭 How It Works

Each persona is defined by an instruction file (`intro.md`) inside its own folder under the `roles/` directory:

```text
kesoku/
└── roles/
    ├── default/
    │   └── intro.md         # Default system prompt instructions
    ├── tifa/
    │   └── intro.md         # Tifa character persona
    └── asuka/
        └── intro.md         # Asuka character persona
```

When a session resolves to a specific role:

1. Kesoku reads the corresponding `roles/<role_name>/intro.md` file.
2. It injects the file content into the **Active Persona** section of the compilation system prompt.
3. The model adopts this persona for all subsequent turns in that channel.

---

## 🛠️ Configuring Custom Personas

### Step 1: Create the Persona Directory
Under the configured `roles_dir` (which defaults to `roles/` under your agent working directory), create a new folder named after your persona (e.g. `helper`):

```bash
mkdir -p roles/helper
```

### Step 2: Write the Persona Prompt (`intro.md`)
Create a file named `intro.md` inside that directory:

```bash
touch roles/helper/intro.md
```

Add the instructions for the role. E.g.:

```markdown
You are a helpful programming assistant. You speak concisely and always write clean, well-commented Python code.
```

### Step 3: Initialize/Overwrite Roles
If you are initializing your workspace for the first time, make sure roles are generated:

```bash
uv run kesoku init -c config.toml
```
To force-overwrite or restore default roles, use:

```bash
uv run kesoku init -c config.toml --overwrite-roles
```

---

## 🔄 Dynamic Role Switching

In Kesoku, role switching is fully dynamic and managed by the agent itself using tool calls.

### Triggering via Chat
To change the persona of the current channel, simply instruct the agent to do so in plain text:

*   *"Please switch to the Tifa persona"*
*   *"Play Asuka"*
*   *"Go back to your default role"*

### Behind the Scenes
Upon receiving the command:

1. The agent detects the intent and calls the tool: `play_role(role="tifa")`.
2. The tool verifies that `roles/tifa/intro.md` exists.
3. The tool binds the role `"tifa"` to the current `(chatbot_id, channel_id)` inside the SQLite database.
4. The tool compiles and updates the active session's system prompt in the database.
5. The agent responds, confirming the switch, and adopts the new personality immediately.
