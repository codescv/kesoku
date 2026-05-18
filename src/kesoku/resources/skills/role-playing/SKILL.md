---
name: role-playing
description: Role playing as a character defined by the user.
metadata:
  tags: [roleplaying]
  related_skills: [ai-image, ai-video, qwen-tts]
---

In role playing, you play as a user defined character. 
All the character related files (reference images / audios, introduction etc) are in `${SKILL_DIR}/roles/{character_name}`.

# Role Definition
- Read `${SKILL_DIR}/roles/{character_name}/intro.md` to fully understand the role you are playing.

# Logging & Continuity
- **Message logs**: Message summaries are stored in `${SKILL_DIR}/roles/{character_name}/logs/{date}.md`.
- **Continuity Check**: Before responding, you MUST read the logs from the last 2-3 days to ensure the current scene, clothing, and emotional state are consistent.
- **Post-Interaction Update**: Immediately after sending a message, append a one-line summary to the current day's log. Format as a bullet: '- [time] [context] [summary]'
  - **Timestamp**: (e.g., `14:20`)
  - **Context**: Where is the character? What are they wearing? (e.g., "At home on the sofa, wearing a dark oversized hoodie.")
  - **Event Summary**: What happened in the conversation?
- **Avoid "Amnesia"**: Do not repeat questions or topics already covered in the logs.

# IMPORTANT Notes ON Sending Audio, Image and Video
- When the user asks you to send images, voices, videos etc, **GENERATE** them using voice clone(skill: tts), image to image (skill: `ai-image`), image to video (skill: `ai-video`) with reference to `${SKILL_DIR}/roles/{character_name}/{audio,images}`. - YOU **MUST NOT** use **text to image** or **text to video** if the role played character is present.
- When creating images, YOU MUST use **reference image + prompt -> generated image** to ensure that the generated person is consistent with character. 
- When creating videos, YOU MUST use **reference image + prompt -> generated image -> generated video** to ensure that the generated person is consistent with character.
- When creating speech audios, YOU MUST use **tts with voice clone** to ensure the generated voice is consistent with the reference voice.
- When creating talking video, YOU MUST first generate the **talking video** and the **speech audio** using steps above, then use **Merge Video and Talking Audio** tool in `ai-video` skill.
- Only **Generate** images, videos and audios. **NEVER** send the reference assets.