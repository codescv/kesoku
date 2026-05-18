# Integration test: AI Image Generation Skill

Use configuration file: `private/config.toml`

# Steps
- Re-Initialize the db and workspace using `kesoku init -w private --overwrite-skills --overwrite-db` to ensure the `ai-image` skill is copied to `private/skills/ai-image`.
- Use `kesoku chat` to ask the agent: "Please check available skills and use the ai image generation skill to generate a picture of a cute cat in a garden. Save it to cat.png."
- Verify that the agent successfully lists skills (`list_skills`), adopts the `ai-image` instructions (`use_skill`), and invokes the shell command using the absolute path header (e.g., `uv run /absolute/path/to/skills/ai-image/scripts/generate_image.py --prompt "cute cat in a garden" --output "cat.png"`).
- Verify that the image `cat.png` is successfully created.
