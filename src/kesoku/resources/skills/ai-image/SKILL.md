---
name: ai-image
description: AI image generation and editing.
metadata:
  tags: [aigc, gcp]
---

tldr: Run `uv run ${SKILL_DIR}/scripts/generate_image.py` Image Generation.

# Maintaining Character Consistency
When generating images for a specific persona or character:
- **Reference Image**: Always provide a high-quality reference photo of the character using the `--image` flag.
- **Prompt Strategy**: In the `--prompt`, focus on describing the *new* environment, lighting, pose, and clothing. The model will use the reference image to maintain the facial features and identity.

# Examples
Below are the most common use cases. For full parameters information, run `uv run ${SKILL_DIR}/scripts/generate_image.py --help`.

Text-to-Image:
`--prompt "A hyperrealistic render of a neon jellyfish floating in a cyber forest" --output "neon_jellyfish.png" --aspect-ratio "9:16"`

Image-to-Image (Character consistency):
`--image "path/to/character_face.jpg" --prompt "The same woman sitting on a cozy sofa at night, wearing a dark oversized hoodie, looking slightly annoyed, cinematic lighting" --output "character_on_sofa.png" --aspect-ratio "3:4"`

# Tip
For errors encountered during API calls, please check the [API reference](https://ai.google.dev/gemini-api/docs/image-generation) to find the correct parameters.