from typing import Literal

from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    """API boundary DTO. Deliberately does NOT expose server-side details
    such as the temporary file path used during inference.
    """

    prediction: Literal["good", "defective"]
    confidence: float = Field(ge=0.0, le=1.0)
    model: str = "resnet50"
    gradcam: str | None = Field(
        default=None,
        description=(
            "Grad-CAM overlay as a 'data:image/png;base64,...' URI. "
            "Present only when the prediction is 'defective'."
        ),
    )


class PredictResponse(BaseModel):
    status: bool = True
    message: str = "OK"
    data: PredictionResult
