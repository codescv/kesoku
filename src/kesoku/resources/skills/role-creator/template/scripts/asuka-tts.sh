#!/bin/bash

# Resolve the absolute path of the workspace
REF_AUDIO="${AWD}/roles/asuka/audio/サイトアスカの写真集ミュージアムを発売を記念した配信の2回目ですね.wav"
REF_TEXT="サイトアスカの写真集ミュージアムを発売を記念した配信の2回目ですね"
GEN_SCRIPT="${AWD}/skills/qwen-tts/scripts/generate_voice.py"

# Help / Usage
if [ $# -lt 2 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Asuka Voice Generator (TTS)"
    echo "Usage: $0 <text> [output_path]"
    echo ""
    echo "Arguments:"
    echo "  text         The Japanese text you want Asuka to speak (preferably Hiragana/Katakana)."
    echo "  output_path  Optional. The output wav file path (default: asuka_output.wav)."
    echo ""
    echo "Example:"
    echo "  $0 \"こんにちは、プロデューサーさん。\" \"asuka_hello.wav\""
    exit 0
fi

TEXT="$1"
OUTPUT="${2}"

echo "Generating voice for Asuka..."
echo "Text:   $TEXT"
echo "Output: $OUTPUT"

# Execute the voice generator using uv run
uv run "$GEN_SCRIPT" \
  --text "$TEXT" \
  --ref-audio "$REF_AUDIO" \
  --ref-text "$REF_TEXT" \
  --output "$OUTPUT"

echo "Voice generation complete: $OUTPUT"
