import base64
import io

import numpy as np
from PIL import Image

from backend.app import inference


def make_image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_good_prediction_has_no_gradcam(monkeypatch):
    calls = {"overlay": 0}

    monkeypatch.setattr(
        inference, "run_prediction", lambda m, d, img: ("good", 0.95)
    )

    def spy_overlay(*args, **kwargs):
        calls["overlay"] += 1
        return np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(inference, "generate_overlay", spy_overlay)

    result = inference.analyze_image(None, None, make_image_bytes())

    assert result.prediction == "good"
    assert result.confidence == 0.95
    assert result.gradcam is None

    assert calls["overlay"] == 0


def test_defective_prediction_returns_gradcam_data_uri(monkeypatch):
    monkeypatch.setattr(
        inference, "run_prediction", lambda m, d, img: ("defective", 0.88)
    )
    monkeypatch.setattr(
        inference,
        "generate_overlay",
        lambda m, d, img: np.full((8, 8, 3), 200, dtype=np.uint8),
    )

    result = inference.analyze_image(None, None, make_image_bytes())

    assert result.prediction == "defective"
    assert result.gradcam is not None
    assert result.gradcam.startswith("data:image/png;base64,")

    # base64 payload must decode into a valid PNG
    payload = result.gradcam.split(",", 1)[1]
    decoded = base64.b64decode(payload)
    with Image.open(io.BytesIO(decoded)) as overlay_img:
        assert overlay_img.format == "PNG"
        assert overlay_img.size == (8, 8)
