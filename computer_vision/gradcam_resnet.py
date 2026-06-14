from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

MODEL_PATH = Path("models/resnet50_bottle.pth")
DATASET_ROOT = Path("processed_dataset/bottle_binary/test/defective")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


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


def main():
    device = get_device()
    print(f"Using device: {device}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    image_path = sorted(DATASET_ROOT.glob("*.png"))[0]
    print(f"Image: {image_path}")

    pil_image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    rgb_image = np.array(pil_image.resize((224, 224))).astype(np.float32) / 255.0

    target_layers = [model.layer4[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=None,
    )[0]

    overlay = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True,
    )

    output_path = OUTPUT_DIR / "gradcam_overlay.png"

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )

    print("Saved:")
    print(output_path)


if __name__ == "__main__":
    main()
