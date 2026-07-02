import io
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from backend.app.constants.uploads import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    MAX_IMAGE_PIXELS,
    MAX_UPLOAD_SIZE_BYTES,
)


def validate_upload(file: UploadFile, file_bytes: bytes) -> None:
    """Validate an uploaded image by extension, MIME type, size and
    decodability. Raises HTTPException on the first failed check; returns
    None when the upload is safe to hand to inference.
    """
    _validate_extension(file.filename)
    _validate_mime_type(file.content_type)
    _validate_size(file_bytes)
    _validate_decodable(file_bytes)


def _validate_extension(filename: str | None) -> None:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file extension '{suffix or 'none'}'. "
                f"Allowed: {sorted(ALLOWED_EXTENSIONS)}."
            ),
        )


def _validate_mime_type(content_type: str | None) -> None:
    # Split off any "; charset=..." parameter before comparing.
    mime = (content_type or "").split(";")[0].strip().lower()
    if mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported content type '{mime or 'none'}'. "
                f"Allowed: {sorted(ALLOWED_MIME_TYPES)}."
            ),
        )


def _validate_size(file_bytes: bytes) -> None:
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"File too large ({len(file_bytes)} bytes). "
                f"Max allowed: {MAX_UPLOAD_SIZE_BYTES} bytes."
            ),
        )


def _validate_decodable(file_bytes: bytes) -> None:
    """Confirm the bytes are a real, decodable image and not a
    decompression bomb. verify() checks structural integrity without
    fully loading pixel data.
    """
    try:
        with Image.open(io.BytesIO(file_bytes)) as image:
            width, height = image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=(
                        f"Image resolution too large ({width}x{height}). "
                        f"Max allowed: {MAX_IMAGE_PIXELS} pixels."
                    ),
                )
            image.verify()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid, decodable image.",
        )
