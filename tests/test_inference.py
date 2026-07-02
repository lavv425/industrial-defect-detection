from pathlib import Path

import pytest

from backend.app import inference


def test_forwards_bytes_and_suffix_then_cleans_up(monkeypatch):
    seen = {}

    def fake_predict(path: str):
        seen["path"] = path
        seen["suffix"] = Path(path).suffix
        seen["exists_during"] = Path(path).exists()
        seen["content"] = Path(path).read_bytes()
        return {"prediction": "good"}

    monkeypatch.setattr(inference, "predict", fake_predict)

    result = inference.predict_uploaded_image(b"raw-bytes", "sample.jpg")

    assert result == {"prediction": "good"}
    assert seen["suffix"] == ".jpg"
    assert seen["exists_during"] is True
    assert seen["content"] == b"raw-bytes"
    # temp file removed after a prediction
    assert not Path(seen["path"]).exists()


def test_defaults_suffix_when_filename_has_none(monkeypatch):
    seen = {}

    def fake_predict(path: str):
        seen["suffix"] = Path(path).suffix
        return {}

    monkeypatch.setattr(inference, "predict", fake_predict)

    inference.predict_uploaded_image(b"x", "no_extension")
    assert seen["suffix"] == ".png"


def test_temp_file_removed_even_when_predict_raises(monkeypatch):
    captured = {}

    def failing_predict(path: str):
        captured["path"] = path
        raise RuntimeError("model blew up")

    monkeypatch.setattr(inference, "predict", failing_predict)

    with pytest.raises(RuntimeError, match="model blew up"):
        inference.predict_uploaded_image(b"x", "sample.png")

    # temp file removed after a prediction
    assert not Path(captured["path"]).exists()
