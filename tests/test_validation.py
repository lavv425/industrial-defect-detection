import io
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, status
from PIL import Image

from backend.app import validation
from backend.app.validation import validate_upload


def make_image_bytes(fmt: str = "PNG", size: tuple[int, int] = (32, 32)) -> bytes:
    """Return the encoded bytes of a small in-memory image."""
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(120, 120, 120)).save(buffer, format=fmt)
    return buffer.getvalue()


def make_upload(filename: str | None, content_type: str | None) -> SimpleNamespace:
    """A minimal stand-in for FastAPI's UploadFile.

    validate_upload only reads `.filename` and `.content_type`, so a
    lightweight stub keeps the tests focused and fast.
    """
    return SimpleNamespace(filename=filename, content_type=content_type)


def test_valid_png_passes():
    file = make_upload("part.png", "image/png")
    # Should not raise and returns None when the upload is safe.
    assert validate_upload(file, make_image_bytes()) is None


def test_valid_jpeg_with_charset_param_passes():
    # MIME types may carry a "; charset=..." suffix that must be stripped.
    file = make_upload("part.jpg", "image/jpeg; charset=binary")
    assert validate_upload(file, make_image_bytes(fmt="JPEG")) is None


def test_rejects_unsupported_extension():
    file = make_upload("malware.exe", "image/png")
    with pytest.raises(HTTPException) as exc:
        validate_upload(file, make_image_bytes())
    assert exc.value.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


def test_rejects_missing_filename():
    file = make_upload(None, "image/png")
    with pytest.raises(HTTPException) as exc:
        validate_upload(file, make_image_bytes())
    assert exc.value.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


def test_rejects_unsupported_mime_type():
    file = make_upload("part.png", "application/octet-stream")
    with pytest.raises(HTTPException) as exc:
        validate_upload(file, make_image_bytes())
    assert exc.value.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


def test_rejects_empty_file():
    file = make_upload("part.png", "image/png")
    with pytest.raises(HTTPException) as exc:
        validate_upload(file, b"")
    assert exc.value.status_code == status.HTTP_400_BAD_REQUEST


def test_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(validation, "MAX_UPLOAD_SIZE_BYTES", 10)
    file = make_upload("part.png", "image/png")
    with pytest.raises(HTTPException) as exc:
        validate_upload(file, make_image_bytes())
    assert exc.value.status_code == status.HTTP_413_CONTENT_TOO_LARGE


def test_rejects_oversized_resolution(monkeypatch):
    # Patch the pixel cap low so a small, cheap image trips it.
    monkeypatch.setattr(validation, "MAX_IMAGE_PIXELS", 100)
    file = make_upload("part.png", "image/png")
    with pytest.raises(HTTPException) as exc:
        validate_upload(file, make_image_bytes(size=(50, 50)))
    assert exc.value.status_code == status.HTTP_413_CONTENT_TOO_LARGE


def test_rejects_non_decodable_bytes():
    # Passes the extension/MIME gate but the payload is not a real image.
    file = make_upload("part.png", "image/png")
    with pytest.raises(HTTPException) as exc:
        validate_upload(file, b"this is definitely not an image")
    assert exc.value.status_code == status.HTTP_400_BAD_REQUEST
