# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "torchaudio",
#     "soundfile",
#     "qwen-tts",
#     "transformers",
#     "accelerate",
#     "sox"
# ]
# ///

import argparse
import os
import re

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def clean_text(text: str) -> str:
    """Remove content within parentheses/brackets.

    e.g. "私（わたし）" -> "私"

    Args:
        text: The input text.

    Returns:
        The cleaned text.
    """
    # Remove () and （）
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"（[^）]*）", "", text)
    # Remove [] and 【】
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"【[^】]*】", "", text)
    return text.strip()


def main() -> None:
    """Main entry point to parse arguments and synthesize voice using Qwen3-TTS."""
    parser = argparse.ArgumentParser(description="Qwen3-TTS Voice Clone Script")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, required=True, help="Output WAV file path")
    parser.add_argument("--ref-audio", type=str, required=True, help="Reference audio file path")
    parser.add_argument("--ref-text", type=str, help="Reference text. Defaults to the filename of ref audio.")

    args = parser.parse_args()

    # Pre-process text: clean parentheses
    input_text = clean_text(args.text)

    # Default ref_text to filename if not provided
    if not args.ref_text:
        args.ref_text = os.path.splitext(os.path.basename(args.ref_audio))[0]
    ref_text = clean_text(args.ref_text)

    print(f"Synthesizing: {input_text}")
    print(f"Using ref audio: {args.ref_audio}")
    print(f"Using ref text: {ref_text}")

    # Initialize model on CPU
    device = "cpu"
    dtype = torch.float32

    print("Loading Qwen3-TTS-12Hz-1.7B-Base model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map=device,
        dtype=dtype,
    )

    # Detect language (simple heuristic or use 'Auto')
    # Since Qwen3 supports 'Auto', we use it.

    print("Generating audio...")
    wavs, sr = model.generate_voice_clone(
        text=input_text,
        language="Auto",
        ref_audio=args.ref_audio,
        ref_text=ref_text,
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    sf.write(args.output, wavs[0], sr)
    print(f"Success! Audio saved to {args.output}")


if __name__ == "__main__":
    main()
