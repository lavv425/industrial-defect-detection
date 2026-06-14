from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

DATASET_ROOT = Path("processed_dataset/bottle_binary")
MODEL_PATH = Path("models/resnet50_bottle.pth")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16


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

    test_dataset = datasets.ImageFolder(
        DATASET_ROOT / "test",
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("Classes:")
    print(test_dataset.class_to_idx)

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)

    print("\nTest Accuracy:")
    print(accuracy)

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=["defective", "good"],
            zero_division=0,
        )
    )

    cm = confusion_matrix(all_labels, all_predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["defective", "good"],
    )

    display.plot()
    plt.title("ResNet50 Confusion Matrix")

    output_path = OUTPUT_DIR / "resnet_confusion_matrix.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print("\nSaved:")
    print(output_path)


if __name__ == "__main__":
    main()
