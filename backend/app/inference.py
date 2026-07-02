import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from computer_vision.gradcam_resnet import generate_overlay
from computer_vision.predict import run_prediction

from backend.app.schemas.prediction import PredictionResult

DEFECTIVE = "defective"


def analyze_image(
    model,
    device: torch.device,
    file_bytes: bytes,
) -> PredictionResult:
    """Run the full inference pipeline on validated image bytes using an
    already-loaded model. Decodes in memory (no temp file), and only computes
    the GradCAM overlay if the sample is defective.
    """
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    prediction, confidence = run_prediction(model, device, image)

    gradcam = None
    if prediction == DEFECTIVE:
        overlay = generate_overlay(model, device, image)
        gradcam = _encode_overlay(overlay)

    return PredictionResult(
        prediction=prediction,
        confidence=confidence,
        gradcam=gradcam,
    )


def _encode_overlay(overlay: np.ndarray) -> str:
    """Encode an RGB uint8 overlay as a PNG data URI."""
    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
