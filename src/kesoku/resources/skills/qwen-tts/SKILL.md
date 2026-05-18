---
name: qwen-tts
description: Text to speech with voice clone using Qwen3-TTS.
metadata:
  tags: [aigc, tts, speech, voice, qwen3]
  platforms: [linux]
---

tldr: Use `uv run ${SKILL_DIR}/scripts/generate_voice.py` to generate speech audio with voice clone.

# Example Usage
```bash
--text "the text you want to synthesize" --ref-audio "/abs/path/to/ref/audio.wav" --ref-text "Ref Audio Transcript" --output "output.wav"
```

# Tips
- `--help` shows the full usage.
- `--ref-text` is optional. If omitted, it defaults to the ref audio filename.