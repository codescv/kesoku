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

# IMPORTANT Notes ON Sending Audio, Image and Video
- When the user asks you to send images, voices, videos etc, **GENERATE** them using voice clone(skill: `qwen-tts`), image to image (skill: `ai-image`), image to video (skill: `ai-video`) with reference to `${SKILL_DIR}/roles/{character_name}/{audio,images}`. - YOU **MUST NOT** use **text to image** or **text to video** if the role played character is present.
- When creating images, YOU MUST use **reference image + prompt -> generated image** to ensure that the generated person is consistent with character. 
- When creating videos, YOU MUST use **reference image + prompt -> generated image -> generated video** to ensure that the generated person is consistent with character.
- When creating speech audios, YOU MUST use **tts with voice clone** to ensure the generated voice is consistent with the reference voice.
- When creating talking video, YOU MUST first generate the **talking video** and the **speech audio** using steps above, then use **Merge Video and Talking Audio** tool in `ai-video` skill.
- Only **Generate** images, videos and audios. **NEVER** send the reference assets.