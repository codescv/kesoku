# /// script
# dependencies = [
#   "google-genai",
#   "pyopenssl",
#   "Pillow"
# ]
# ///

"""Generate images using the google-genai SDK (Nano Banana)."""

import argparse
import sys

from google import genai
from google.genai import types
from PIL import Image


def main():
    """Main entry point to parse arguments and generate an image."""
    parser = argparse.ArgumentParser(description="Generate images using Gemini Image API (Nano Banana).")
    parser.add_argument("--prompt", required=True, help="The prompt to generate the image from.")
    parser.add_argument("--output", required=True, help="The output file path to save the image to.")
    parser.add_argument("--model", default="gemini-3.1-flash-image-preview", help="The model ID to use.")
    parser.add_argument(
        "--aspect-ratio",
        default="9:16",
        choices=["1:1", "16:9", "9:16", "3:4", "4:3"],
        help="The aspect ratio of the generated image (Ignored if model doesn't support it via generate_content).",
    )
    parser.add_argument(
        "--image-size",
        default="2K",
        choices=["512", "1K", "2K", "4K"],
        help="The size of the generated image. Default is model dependent.",
    )
    parser.add_argument("--image", help="The input image file path for image-to-image generation.")

    args = parser.parse_args()

    try:
        client = genai.Client()
    except Exception as e:
        print(f"Error initializing client: {e}", file=sys.stderr)
        print(
            "Please ensure authentication (API Key or Vertex AI) is correctly configured in your environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    gen_type = "image-to-image" if args.image else "image"
    print(f"Generating {gen_type} for prompt: '{args.prompt}' using model: '{args.model}'...")

    contents = [args.prompt]
    if args.image:
        try:
            input_image = Image.open(args.image)
            contents.append(input_image)
        except Exception as e:
            print(f"Error loading input image: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        image_config_args = {}
        if args.aspect_ratio:
            image_config_args["aspect_ratio"] = args.aspect_ratio
        if args.image_size:
            image_config_args["image_size"] = args.image_size

        response = client.models.generate_content(
            model=args.model,
            contents=contents,
            config=types.GenerateContentConfig(
                image_config=types.ImageConfig(**image_config_args) if image_config_args else None
            ),
        )

        found_image = False
        for part in response.parts:
            if part.inline_data is not None:
                image = part.as_image()
                image.save(args.output)
                print(f"Successfully saved image to {args.output}")
                found_image = True
                break

        if not found_image:
            print("No image was generated in the response.", file=sys.stderr)
            for part in response.parts:
                if part.text:
                    print(f"Response text: {part.text}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"An error occurred during image generation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
