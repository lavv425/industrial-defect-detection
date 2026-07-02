from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from computer_vision.predict import TRANSFORM, load_model

DATASET_ROOT = Path("processed_dataset/bottle_binary/test/defective")
OUTPUT_DIR = Path("outputs")


def generate_overlay(model, device: torch.device, image: Image.Image) -> np.ndarray:
    """Compute the Grad-CAM heatmap for an already-loaded model and return it
    blended over the (224x224) input as an RGB uint8 array.

    Note: this runs a backward pass and is far more expensive than a plain
    prediction — only call it when you actually need the explanation.
    """
    input_tensor = TRANSFORM(image).unsqueeze(0).to(device)

    rgb_image = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

    target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

    return show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)


def main():
    model, device = load_model()
    print(f"Using device: {device}")

    image_path = sorted(DATASET_ROOT.glob("*.png"))[0]
    print(f"Image: {image_path}")

    pil_image = Image.open(image_path).convert("RGB")
    overlay = generate_overlay(model, device, pil_image)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "gradcam_overlay.png"

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )

    print("Saved:")
    print(output_path)


if __name__ == "__main__":
    main()
