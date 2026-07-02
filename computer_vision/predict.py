from pathlib import Path
import sys

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image


MODEL_PATH = Path("models/resnet50_bottle.pth")

IDX_TO_CLASS = {
    0: "defective",
    1: "good",
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_model():
    model = models.resnet50(weights=None)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2),
    )

    return model


def load_model(model_path: Path = MODEL_PATH, device: torch.device | None = None):
    """Build the network, load the trained weights and put it in eval mode.

    Call this ONCE and reuse the returned model - loading the checkpoint is
    expensive and must not happen on every request.
    """
    device = device or get_device()

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, device


def run_prediction(model, device: torch.device, image: Image.Image) -> tuple[str, float]:
    """Run a forward pass on an already-loaded model and return
    (prediction, confidence). No disk I/O, no checkpoint loading.
    """
    input_tensor = TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    prediction = IDX_TO_CLASS[int(predicted_idx.item())]

    return prediction, float(confidence.item())


def predict(image_path: str):
    """Self-contained one-shot prediction for CLI use. Loads the model each
    call - fine for a script, not for the API (see load_model docstring).
    """
    model, device = load_model()
    image = Image.open(image_path).convert("RGB")
    prediction, confidence = run_prediction(model, device, image)

    return {
        "image": image_path,
        "prediction": prediction,
        "confidence": confidence,
        "model": "resnet50",
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python computer_vision/predict.py path/to/image.png"
        )

    result = predict(sys.argv[1])

    print(result)
