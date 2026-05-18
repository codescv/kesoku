# /// script
# dependencies = [
#   "google-genai",
#   "requests",
#   "pyopenssl"
# ]
# ///

"""
Generate videos using the google-genai SDK (Veo).
"""

import os
import sys
import argparse
import time
from google import genai
from google.genai import types

def main():
    parser = argparse.ArgumentParser(description="Generate video using Google Veo")
    parser.add_argument("--prompt", required=True, help="Text prompt for the video")
    parser.add_argument("--output", required=True, help="Output MP4 filename")
    parser.add_argument("--image", help="Optional reference/starting image")
    parser.add_argument("--model", default="veo-3.1-generate-001", help="Veo model to use (default: veo-3.1-generate-001)")
    parser.add_argument("--aspect-ratio", help="Aspect ratio for the generated video (e.g., '16:9', '9:16')")
    args = parser.parse_args()

    print(f"Initializing Veo generation for: {args.output}")
    try:
        client = genai.Client()
        
        model_name = args.model
        
        # Prepare configuration
        config_kwargs = {}
        if args.aspect_ratio:
            config_kwargs["aspect_ratio"] = args.aspect_ratio
        
        config = types.GenerateVideosConfig(**config_kwargs) if config_kwargs else None
        
        print(f"Requesting video generation from {model_name}...")
        if args.image and os.path.exists(args.image):
            import mimetypes
            from google.genai.types import Image
            
            print(f"Loading reference image: {args.image}")
            with open(args.image, "rb") as image_file:
                image_bytes = image_file.read()
            
            mime_type, _ = mimetypes.guess_type(args.image)
            if not mime_type:
                mime_type = "image/png"
            
            image_input = Image(
                image_bytes=image_bytes,
                mime_type=mime_type
            )
            
            print(f"Requesting video generation from {model_name}...")
            operation = client.models.generate_videos(
                model=model_name,
                prompt=args.prompt,
                image=image_input,
                config=config
            )
        else:
            operation = client.models.generate_videos(
                model=model_name,
                prompt=args.prompt,
                config=config
            )
        
        print(f"Video generation initiated. Operation name: {operation.name}")
        
        print("Waiting for generation to complete (this may take several minutes)...")
        while not operation.done:
            time.sleep(20)
            operation = client.operations.get(operation)
            print(f"Operation status: {'Done' if operation.done else 'In progress'}")
            
        print("\nGeneration complete!")
        
        if operation.error:
            print(f"Error generating video: {operation.error}")
            sys.exit(1)
            
        if operation.response and operation.response.generated_videos:
            video_obj = operation.response.generated_videos[0].video
            
            try:
                if hasattr(video_obj, "video_bytes") and video_obj.video_bytes:
                    print(f"Saving video from video_bytes ({len(video_obj.video_bytes)} bytes)...")
                    with open(args.output, "wb") as f:
                        f.write(video_obj.video_bytes)
                    print(f"Video saved: {os.path.abspath(args.output)}")
                    print(f"MEDIA:{os.path.abspath(args.output)}")
                elif hasattr(video_obj, "save"):
                    print(f"Saving video using save() method...")
                    video_obj.save(args.output)
                    print(f"Video saved: {os.path.abspath(args.output)}")
                    print(f"MEDIA:{os.path.abspath(args.output)}")
                else:
                    # Use client.files.download to get the video bytes as fallback
                    print(f"Attempting standard download for URI: {video_obj.uri}")
                    video_bytes = client.files.download(file=video_obj)
                    with open(args.output, "wb") as f:
                        f.write(video_bytes)
                    print(f"Video saved: {os.path.abspath(args.output)}")
                    print(f"MEDIA:{os.path.abspath(args.output)}")
            except Exception as dl_err:
                print(f"Download failed: {dl_err}. Trying fallback...")
                try:
                    import requests
                    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GENAI_API_KEY")
                    if not api_key:
                        raise ValueError("No API key found in environment")
                    
                    if video_obj.uri:
                        download_url = f"{video_obj.uri}?key={api_key}"
                        print(f"Downloading from URI with key: {video_obj.uri}")
                        res = requests.get(download_url)
                        res.raise_for_status()
                        with open(args.output, "wb") as f:
                            f.write(res.content)
                        print(f"Video saved via fallback: {os.path.abspath(args.output)}")
                        print(f"MEDIA:{os.path.abspath(args.output)}")
                    else:
                        raise ValueError("No URI available for fallback download")
                except Exception as fallback_err:
                    print(f"Fallback download also failed: {fallback_err}")
                    sys.exit(1)
        else:
            print("No video found in the operation response.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()