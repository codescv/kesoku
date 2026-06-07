#!/bin/bash

# Resolve the absolute path of the workspace
REF_IMAGE="${AWD}/roles/asuka/images/asuka.jpg"
GEN_SCRIPT="${AWD}/skills/ai-image/scripts/generate_image.py"

# Help / Usage
if [ $# -lt 2 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Asuka Character Image Generator (Image-to-Image)"
    echo "Usage: $0 <prompt> [output_path] [additional_options...]"
    echo ""
    echo "Arguments:"
    echo "  prompt              The prompt describing the new environment, pose, clothing, etc."
    echo "  output_path         Optional. The output png file path (default: asuka_output.png)."
    echo "  additional_options  Optional. Additional flags for generate_image.py (e.g. --aspect-ratio 3:4)."
    echo ""
    echo "Example:"
    echo "  $0 \"The same girl sitting on a cozy sofa at night, wearing a dark oversized hoodie, cinematic lighting\" \"asuka_sofa.png\" --aspect-ratio 3:4"
    exit 0
fi

PROMPT="$1"
OUTPUT="$2"
shift 2

echo "Generating character image for Asuka..."
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT"

# Execute the image generator using uv run
uv run "$GEN_SCRIPT" \
  --image "$REF_IMAGE" \
  --prompt "$PROMPT" \
  --output "$OUTPUT" \
  "$@"

echo "Image generation complete: $OUTPUT"
