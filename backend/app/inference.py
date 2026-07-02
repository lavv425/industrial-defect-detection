import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from computer_vision.predict import predict


def predict_uploaded_image(file_bytes: bytes, filename: str):
    suffix = Path(filename).suffix or ".png"

    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        return predict(temp_path)
    finally:
        # remove temp file, no mem leak
        Path(temp_path).unlink(missing_ok=True)
