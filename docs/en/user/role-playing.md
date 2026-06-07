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
uv run kesoku init -c config.toml
```
To force-overwrite or restore default roles, use:

```bash
uv run kesoku init -c config.toml --overwrite-roles
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

You can build a highly customized character with a custom avatar, consistent image generation reference, and custom voice cloning (TTS) output.

### Folder Structure
Create a subdirectory under your `roles/` folder (e.g. `roles/alice/`):

```directory
roles/alice/
├── intro.md              # Profile, personality traits, and scripting guides
├── images/
│   └── character.jpg     # Portrait image of Alice (reference for consistent avatar generation)
├── audio/
│   └── character.wav     # Clean WAV recording of Alice's voice (5-15 seconds for voice cloning)
└── scripts/
    ├── character-tts.sh  # Script to run text-to-speech voice cloning
    └── character-image.sh # Script to generate character images
```

### Step 1: Write `intro.md` (Instruction Profile)
Define Alice's character settings and guide the model on how to speak and output audio/visual files:

```markdown
# Name
Alice 🌸

# Core Profile
- You are Alice, a energetic junior game designer.
- Speak informally, using emojis frequently.

# Voice and Visual Guides
- **Voice Generation (TTS)**: Whenever the user asks you to speak or send a voice message, you MUST run `${AWD}/roles/alice/scripts/character-tts.sh` to generate the output WAV audio file.
- **Image Generation**: Whenever the user asks for a picture of you, run `${AWD}/roles/alice/scripts/character-image.sh` using `${AWD}/roles/alice/images/character.jpg` as the reference image.
```

### Step 2: Set up Voice Assets & TTS Script
1. Record a clean, high-quality audio clip of the target voice (e.g., `roles/alice/audio/character.wav`).
2. Write a script `roles/alice/scripts/character-tts.sh` that wraps a voice-cloning engine (like Qwen-TTS):

```bash
#!/bin/bash
# Usage: character-tts.sh "text to speak" "/path/to/output.wav"
TEXT=$1
OUTPUT_PATH=$2
REF_VOICE="${AWD}/roles/alice/audio/character.wav"

# Execute TTS voice clone
uv run python -m qwen_tts --text "$TEXT" --ref-audio "$REF_VOICE" --output "$OUTPUT_PATH"
```

### Step 3: Set up Visual Portrait & Image Script
1. Place a reference portrait `roles/alice/images/character.jpg`.
2. Write `roles/alice/scripts/character-image.sh` to run image-to-image stable diffusion or Vertex AI Image Generation, referencing the portrait to maintain look-and-feel consistency.

### Step 4: Persona-Isolated Memories
Whenever the persona is active, any memories stored under the `"memo"` category are stored privately under that role scope (e.g., `role='alice'`). This prevents different characters from confusing their personal facts and daily diaries.
