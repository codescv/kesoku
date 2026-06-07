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
    ├── coder/
    │   └── intro.md         # Coder character persona
    └── helper/
        └── intro.md         # Helper character persona
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
kesoku init -c config.toml
```
To force-overwrite or restore default roles, use:

```bash
kesoku init -c config.toml --overwrite-roles
```

---

## 🔄 Dynamic Role Switching

In Kesoku, role switching is fully dynamic and can be managed by slash commands or by the agent itself.

### 1. Using Slash Commands
On chat platforms (such as Discord or WeChat), you can manage personas instantly:
*   `/role`: List all active and available personas in the workspace.
*   `/role {name}`: Switch the current channel or thread to the specified persona (e.g., `/role coder`).

> [!TIP]
> **Discord Thread Inheritance**: In Discord, active threads inherit the parent channel's persona by default, but you can run `/role {name}` inside the thread to bind a distinct persona.

### 2. Triggering via Chat Dialog
You can also ask the agent to change its persona in plain conversational text:
*   *"Please switch to the coder persona"*
*   *"Play helper"*
*   *"Go back to your default role"*

Behind the scenes, the agent detects the intent, calls `play_role(role="coder")`, binds the new role in SQLite, and dynamically rebuilds the system prompt.

---

## 🎨 Tutorial: Creating a Vivid Character Persona (Voice Clone & Visual Avatar)

In Kesoku, you don't need to manually create complex role directories or write shell scripts. Kesoku has a built-in **`role-creator` skill** that allows the agent to interactively design, create, and automatically bootstrap new character personas for you!

### 🚀 Creating a Persona Interactively

Simply ask the agent in your chat:
> *"Help me design a new character named Alice"*
> or
> *"Start the role creator tool"*

The agent will activate the `role-creator` skill and guide you through the process step-by-step:
1. **Core Concept**: Ask for the name, character traits, linguistic style, and catchphrases.
2. **Avatar Sample (Optional)**: Help you select or upload a reference portrait image for consistent visual generations.
3. **Voice Sample (Optional)**: Help you provide a 5-15 seconds WAV clip of the target voice along with its transcript to clone a matching TTS voice output.

Once gathered, the agent automatically generates the folder structures, configs, and custom rendering scripts.

---

### 📂 Generated Folder Structure

Once `role-creator` runs, it scaffolds the following structure under your `roles/` directory (using `roles/alice/` as an example):

```directory
roles/alice/
├── intro.md              # Profile, personality traits, and scripting guides (auto-generated)
├── images/
│   └── character.jpg     # Reference portrait image of Alice (if provided, for consistent visual avatar generation)
├── audio/
│   └── character.wav     # Reference WAV audio clip of Alice (if provided, for voice cloning)
└── scripts/
    ├── alice-tts.sh      # Executable script to generate voice cloned TTS (Hiragana/Katakana for Japanese)
    └── alice-image.sh    # Executable script to generate consistent illustrations of Alice
```

---

### 🔄 Persona-Isolated Memories

Whenever the persona is active, any memories stored under the `"memo"` category are stored privately under that role scope (e.g., `role='alice'`). This prevents different characters from confusing their personal facts and daily diaries.
