from pathlib import Path
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from sklearn.metrics import classification_report, accuracy_score

DATASET_ROOT = Path("processed_dataset/bottle_binary")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main():
    device = get_device()
    print(f"Using device: {device}")

    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(
        DATASET_ROOT / "train",
        transform=train_transforms,
    )

    val_dataset = datasets.ImageFolder(
        DATASET_ROOT / "val",
        transform=eval_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("Classes:")
    print(train_dataset.class_to_idx)

    weights = models.ResNet50_Weights.DEFAULT

    model = models.resnet50(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 2),
    )

    model = model.to(device)

    class_counts = torch.bincount(torch.tensor(train_dataset.targets))

    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    print("Class counts:")
    print(class_counts)

    print("Class weights:")
    print(class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=LR,
    )

    best_accuracy = 0.0
    best_model_state = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()

        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)

        print(f"Train loss: {train_loss / len(train_loader):.4f}")
        print(f"Val accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": train_dataset.class_to_idx,
            "best_val_accuracy": best_accuracy,
        },
        MODEL_DIR / "resnet50_bottle.pth",
    )

    print("\nBest validation accuracy:")
    print(best_accuracy)

    print("\nValidation Classification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=["defective", "good"],
            zero_division=0,
        )
    )

    print("\nModel saved:")
    print(MODEL_DIR / "resnet50_bottle.pth")


if __name__ == "__main__":
    main()
