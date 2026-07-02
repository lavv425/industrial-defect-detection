import io

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import main
from backend.app.constants.routes import HEALT_ROUTE, PREDICT_ROUTE


def make_image_bytes(fmt: str = "PNG", size: tuple[int, int] = (32, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(120, 120, 120)).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def client() -> TestClient:
    return TestClient(main.app)


@pytest.fixture
def mock_inference(monkeypatch):
    """
    Replace the real (ResNet-backed) inference with a stub
    """
    calls = {}

    def fake_predict(file_bytes: bytes, filename: str):
        calls["file_bytes"] = file_bytes
        calls["filename"] = filename
        return {
            "image": filename,
            "prediction": "good",
            "confidence": 0.99,
            "model": "resnet50",
        }

    monkeypatch.setattr(main, "predict_uploaded_image", fake_predict)
    return calls


def test_health(client):
    response = client.get(HEALT_ROUTE)
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": True, "message": "API up n'running"}


def test_predict_success(client, mock_inference):
    image = make_image_bytes()
    response = client.post(
        PREDICT_ROUTE,
        files={"file": ("part.png", image, "image/png")},
    )

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["status"] is True
    assert body["data"]["prediction"] == "good"
    assert body["data"]["model"] == "resnet50"

    # endpoint forwarded the exact bytes and filename to inference
    assert mock_inference["file_bytes"] == image
    assert mock_inference["filename"] == "part.png"


def test_predict_rejects_unsupported_extension(client, mock_inference):
    response = client.post(
        PREDICT_ROUTE,
        files={"file": ("malware.exe", make_image_bytes(), "image/png")},
    )
    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    # validation must fail before inference is reached.
    assert mock_inference == {}


def test_predict_rejects_empty_file(client, mock_inference):
    response = client.post(
        PREDICT_ROUTE,
        files={"file": ("part.png", b"", "image/png")},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert mock_inference == {}


def test_predict_rejects_non_decodable_bytes(client, mock_inference):
    response = client.post(
        PREDICT_ROUTE,
        files={"file": ("part.png", b"not an image", "image/png")},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert mock_inference == {}


def test_predict_requires_file(client):
    response = client.post(PREDICT_ROUTE)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
