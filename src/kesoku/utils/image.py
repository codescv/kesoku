"""Image processing utility functions for Kesoku AI Agent framework."""

import io
import logging

from PIL import Image

logger = logging.getLogger(__name__)


def detect_image_mime_type(file_bytes: bytes, fallback_mime: str = "image/jpeg") -> tuple[str, str]:
    """Detect image mime type and matching file extension from magic bytes.

    Args:
        file_bytes: Raw image bytes.
        fallback_mime: Fallback MIME type if magic bytes don't match.

    Returns:
        A tuple of (mime_type, extension_with_dot).
    """
    if file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", ".png"
    if file_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", ".jpg"
    if file_bytes.startswith(b"GIF87a") or file_bytes.startswith(b"GIF89a"):
        return "image/gif", ".gif"
    if file_bytes.startswith(b"RIFF") and len(file_bytes) >= 12 and file_bytes[8:12] == b"WEBP":
        return "image/webp", ".webp"

    ext = ".jpg"
    if fallback_mime == "image/png":
        ext = ".png"
    elif fallback_mime == "image/gif":
        ext = ".gif"
    elif fallback_mime == "image/webp":
        ext = ".webp"
    return fallback_mime, ext


def compress_image(data: bytes, max_size: int = 1024 * 1024) -> bytes:
    """Compress an image to be under max_size bytes.

    Converts the image to RGB and saves it as a JPEG with compression. Resizes
    the dimensions if the file size remains too large after initial compression.

    Args:
        data: The original image bytes.
        max_size: The target maximum file size in bytes (default 1MB).

    Returns:
        The compressed image bytes, or original bytes if compression fails.
    """
    try:
        img = Image.open(io.BytesIO(data))

        # Convert RGBA or Palette (P) images to RGB for JPEG compatibility
        if img.mode in ("RGBA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "RGBA":
                background.paste(img, mask=img.split()[3])
            else:
                background.paste(img)
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # Try initial saving with high quality
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=85)
        compressed = out.getvalue()

        # Loop to downscale/compress iteratively until it fits max_size
        attempts = 0
        while len(compressed) > max_size and attempts < 4:
            attempts += 1
            w, h = img.size
            # Scale down by 0.7x each time to aggressively reduce resolution
            img = img.resize((int(w * 0.7), int(h * 0.7)), Image.Resampling.LANCZOS)

            # Reduce quality iteratively
            quality = max(30, 75 - attempts * 10)
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=quality)
            compressed = out.getvalue()

        return compressed
    except Exception as exc:
        logger.warning("Image utility: failed to compress image, sending original: %s", exc)
        return data
