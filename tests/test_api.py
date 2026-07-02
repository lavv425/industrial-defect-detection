import io

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import main
from backend.app.constants.routes import API_PREFIX, HEALTH_ROUTE, PREDICT_ROUTE
from backend.app.schemas.prediction import PredictionResult

HEALTH_URL = API_PREFIX + HEALTH_ROUTE
PREDICT_URL = API_PREFIX + PREDICT_ROUTE


def make_image_bytes(fmt: str = "PNG", size: tuple[int, int] = (32, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(120, 120, 120)).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def client() -> TestClient:
    main.app.dependency_overrides[main.get_model] = lambda: (None, None)
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


@pytest.fixture
def mock_inference(monkeypatch):
    """Replace the ResNet-backed inference with a stub. Records the call and
    returns a canned PredictionResult so API tests stay fast/deterministic.
    """
    calls = {}
    result = PredictionResult(prediction="good", confidence=0.99)

    def fake_analyze(model, device, file_bytes: bytes) -> PredictionResult:
        calls["file_bytes"] = file_bytes
        return calls.setdefault("result", result)

    monkeypatch.setattr(main, "analyze_image", fake_analyze)
    return calls


def test_health(client):
    response = client.get(HEALTH_URL)
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": True, "message": "API up n'running"}


def test_predict_good_has_no_gradcam(client, mock_inference):
    image = make_image_bytes()
    response = client.post(
        PREDICT_URL,
        files={"file": ("part.png", image, "image/png")},
    )

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["status"] is True
    assert body["message"] == "OK"
    assert body["data"]["prediction"] == "good"
    assert body["data"]["model"] == "resnet50"
    assert body["data"]["gradcam"] is None

    # endpoint forwarded the exact bytes to inference
    assert mock_inference["file_bytes"] == image


def test_predict_defective_returns_gradcam(client, monkeypatch):
    def fake_analyze(model, device, file_bytes):
        return PredictionResult(
            prediction="defective",
            confidence=0.87,
            gradcam="data:image/png;base64,ZmFr ==",
        )

    monkeypatch.setattr(main, "analyze_image", fake_analyze)

    response = client.post(
        PREDICT_URL,
        files={"file": ("part.png", make_image_bytes(), "image/png")},
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()["data"]
    assert data["prediction"] == "defective"
    assert data["gradcam"].startswith("data:image/png;base64,")


def test_predict_rejects_unsupported_extension(client, mock_inference):
    response = client.post(
        PREDICT_URL,
        files={"file": ("malware.exe", make_image_bytes(), "image/png")},
    )
    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    # validation must fail before inference is reached.
    assert mock_inference == {}


def test_predict_rejects_empty_file(client, mock_inference):
    response = client.post(
        PREDICT_URL,
        files={"file": ("part.png", b"", "image/png")},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert mock_inference == {}


def test_predict_rejects_non_decodable_bytes(client, mock_inference):
    response = client.post(
        PREDICT_URL,
        files={"file": ("part.png", b"not an image", "image/png")},
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert mock_inference == {}


def test_predict_requires_file(client):
    response = client.post(PREDICT_URL)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
