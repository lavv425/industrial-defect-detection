from pathlib import Path
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
)

DATASET_ROOT = Path("processed_dataset/bottle_binary")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_images(split: str):
    X = []
    y = []

    classes = {
        "good": 0,
        "defective": 1,
    }

    for class_name, label in classes.items():
        class_dir = DATASET_ROOT / split / class_name

        for image_path in class_dir.glob("*.png"):
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (128, 128))

            features = hog(
                image,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
            )

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


def main():
    model = joblib.load(MODEL_DIR / "hog_svm.pkl")
    scaler = joblib.load(MODEL_DIR / "hog_scaler.pkl")

    X_test, y_test = load_images("test")
    X_test = scaler.transform(X_test)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Test Accuracy:")
    print(accuracy)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["good", "defective"],
        )
    )

    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["good", "defective"],
    )

    display.plot()

    plt.title("HOG + SVM Confusion Matrix")
    plt.savefig(
        OUTPUT_DIR / "hog_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    print("\nSaved:")
    print(OUTPUT_DIR / "hog_confusion_matrix.png")


if __name__ == "__main__":
    main()
