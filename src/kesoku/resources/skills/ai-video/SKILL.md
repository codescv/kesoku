---
name: ai-video
description: AI video generation and editing.
metadata:
  tags: [aigc]
  related_skills: [ai-image, qwen-tts]
---

This skill is for video generation and editing. Every section is optional; use only what you need.

# Video Generation

Run `uv run ${SKILL_DIR}/scripts/generate_video.py` for Video Generation.
For full usage, run `uv run ${SKILL_DIR}/scripts/generate_video.py --help`.

## Examples Usage
Generate text-to-video: 
`--prompt "A cyberpunk city at night with neon lights" --output "output.mp4"`

Generate image-to-video (animate an image):
`--prompt "Make the character speak and smile" --image "/path/to/image.png" --output "output.mp4"`


# Merging Video and Speech
When making talking videos, it's useful to loop the video and merge the audio with this command:

```bash
ffmpeg -y -stream_loop -1 -i "<video_path>" -i "<speech_audio_path>" -c:v libx264 -crf 28 -preset fast -c:a aac -map 0:v:0 -map 1:a:0 -shortest -fflags +genpts "<output_video_path>"
```