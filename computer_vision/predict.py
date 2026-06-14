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


def predict(image_path: str):
    device = get_device()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    prediction = IDX_TO_CLASS[int(predicted_idx.item())]

    return {
        "image": image_path,
        "prediction": prediction,
        "confidence": float(confidence.item()),
        "model": "resnet50",
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python computer_vision/predict.py path/to/image.png"
        )

    result = predict(sys.argv[1])

    print(result)