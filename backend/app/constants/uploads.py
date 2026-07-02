ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
)

ALLOWED_MIME_TYPES: frozenset[str] = frozenset(
    {"image/png", "image/jpeg", "image/bmp", "image/webp"}
)

MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024

# Max decoded pixel count
MAX_IMAGE_PIXELS: int = 24_000_000  # 24MP
