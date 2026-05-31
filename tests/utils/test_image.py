"""Unit tests for the image processing utility functions."""

import io

from PIL import Image

from kesoku.utils.image import compress_image, detect_image_mime_type


def test_detect_image_mime_type() -> None:
    """Test detecting image MIME types and extensions from magic bytes."""
    # PNG magic bytes: \x89PNG\r\n\x1a\n
    assert detect_image_mime_type(b"\x89PNG\r\n\x1a\n") == ("image/png", ".png")

    # JPEG magic bytes: \xff\xd8\xff
    assert detect_image_mime_type(b"\xff\xd8\xff") == ("image/jpeg", ".jpg")

    # GIF magic bytes: GIF89a or GIF87a
    assert detect_image_mime_type(b"GIF89a") == ("image/gif", ".gif")
    assert detect_image_mime_type(b"GIF87a") == ("image/gif", ".gif")

    # WEBP magic bytes: RIFFxxxxWEBP
    assert detect_image_mime_type(b"RIFF\x00\x00\x00\x00WEBP") == ("image/webp", ".webp")

    # Fallbacks
    assert detect_image_mime_type(b"random_data", fallback_mime="image/png") == ("image/png", ".png")
    assert detect_image_mime_type(b"random_data", fallback_mime="image/gif") == ("image/gif", ".gif")
    assert detect_image_mime_type(b"random_data", fallback_mime="image/webp") == ("image/webp", ".webp")
    assert detect_image_mime_type(b"random_data") == ("image/jpeg", ".jpg")


def test_compress_image() -> None:
    """Test compressing a large image under the size threshold."""
    # Create a large 2000x2000 solid Red RGBA image (will be compressed to JPEG RGB)
    img = Image.new("RGBA", (2000, 2000), color=(255, 0, 0, 255))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    raw_bytes = img_byte_arr.getvalue()

    # Set limit to 50KB (very small, will force aggressive compression and downscaling)
    limit = 50 * 1024
    compressed_bytes = compress_image(raw_bytes, max_size=limit)

    # Verify that it is under the limit and format changed to JPEG
    assert len(compressed_bytes) <= limit
    # Verify it detects as JPEG now
    mime, ext = detect_image_mime_type(compressed_bytes)
    assert mime == "image/jpeg"

    # Also verify that we can open the compressed image successfully
    compressed_img = Image.open(io.BytesIO(compressed_bytes))
    assert compressed_img.size[0] < 2000  # Verify it was downscaled
    assert compressed_img.size[1] < 2000
