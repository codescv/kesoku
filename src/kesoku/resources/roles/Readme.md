# Kesoku Persona Roleplay Guide 🎭

Welcome to the **Kesoku built-in Persona and Role-playing System**! This system allows you to bind custom personas to any conversation channel (Discord channels/threads, WeChat rooms, etc.), defining how the agent behaves, speaks, and dynamically generates consistent voice and image outputs.

---

## 📂 Directory Structure

Personas are located in the `roles/` directory under the **Agent Working Directory (AWD)**. Each character must have its own folder with the following structure:

```directory
roles/
├── default/                  # The fallback global assistant persona
│   └── intro.md
├── {character_name}/         # E.g., 'asuka'
│   ├── intro.md              # Profile, personality, and guidelines (Markdown)
│   ├── images/               # Reference images of the character for visual consistency
│   │   └── character.jpg     # (JPEG/PNG format)
│   ├── audio/                # Reference audio snippets for voice clone consistency
│   │   └── character.wav     # (PCM WAV format)
│   └── scripts/              # Character-specific generation scripts (Optional)
│       ├── {character}-tts.sh
│       └── {character}-image.sh
```

---

## 🎨 Step-by-Step: Defining a Custom Persona

### 1. Setup Character Profile (`intro.md`)
Create an `intro.md` inside your character folder. This file defines the core personality, linguistic quirks, background, and instructs the LLM on how to invoke character-specific scripts:

```markdown
# Name
Cyber Character 🚀

# Core Truths
- Describe how the character behaves (e.g., "Helpful but acts tsundere", "Speaks informally").
- Define language preferences (e.g., "Teaches Japanese, translates answers into Chinese").

# Script Generation Guides
- **Voice Generation (TTS)**: You must run `${AWD}/roles/{character_name}/scripts/character-tts.sh` to produce a WAV voice.
- **Image Generation**: You must run `${AWD}/roles/{character_name}/scripts/character-image.sh` with reference to `${AWD}/roles/{character_name}/images/character.jpg`.
```

### 2. Add Custom Visual Assets (`images/`)
To ensure that the generated images/videos look exactly like your character:
- Create the `images/` subdirectory.
- Place high-quality portrait images of the character here (e.g., `character.jpg`).
- When the agent is requested to send a picture, it will pass this image as a reference along with its generation prompt to the `ai-image` skill to perform **Image-to-Image** generation.

### 3. Add Custom Voice Assets (`audio/`)
To ensure voice-cloning consistency:
- Create the `audio/` subdirectory.
- Place a clear, noise-free WAV audio recording of the character's voice here (typically 5-15 seconds long).
- The `qwen-tts` or custom voice cloning skills will use this WAV file as a reference target to clones the character's speech output precisely.

### 4. Define Custom Generation Scripts (`scripts/` - Optional)
For advanced characters, you can provide custom wrapper scripts inside the `scripts/` folder to run localized generation pipelines.

#### Custom TTS script (`scripts/asuka-tts.sh` Example):
```bash
#!/bin/bash
# Usage: asuka-tts.sh "text to speak" "/path/to/output.wav"
TEXT=$1
OUTPUT_PATH=$2
REF_VOICE="${AWD}/roles/asuka/audio/asuka.wav"

# Run TTS with voice cloning
uv run python -m qwen_tts --text "$TEXT" --ref-audio "$REF_VOICE" --output "$OUTPUT_PATH"
```

---

## 🚀 Commands and Usage

### 1. Switching Personas in Chat Channels
Use the `/role` command inside your chat platforms (Discord or WeChat) to manage personas:

- `/role`: Display the current active persona and a list of all available personas.
- `/role {name}`: Switch the active persona for the current channel/thread instantly (e.g., `/role asuka`).

> [!TIP]
> **Discord Thread Inheritance**: In Discord, active threads automatically inherit their parent channel's persona binding by default, but you can run `/role {name}` inside the thread to bind a distinct persona!

### 2. Dynamic Agent Persona Switching (`play_role` tool)
The agent itself can switch its persona dynamically based on user prompt requirements by calling:
- `play_role(role="character_name")`

---

## 🧠 Memory Isolation (`fun_fact` category)

To ensure that different characters do not confuse their interactions:
- Globally shared knowledge categories (`progress`, `user_profile`, `learnings`) are always stored in the globally shared `"default"` role scope.
- The `"fun_fact"` memory category is role-isolated. Whenever the agent is active under a specific persona, facts and memorable daily events are stored privately under that character's role scope (e.g., `role='asuka'`).
