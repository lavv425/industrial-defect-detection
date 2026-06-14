from pathlib import Path
import cv2
import joblib
import numpy as np

from skimage.feature import hog

from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from sklearn.preprocessing import StandardScaler

DATASET_ROOT = Path("processed_dataset/bottle_binary")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def load_images(split: str):
    X = []
    y = []

    split_dir = DATASET_ROOT / split

    classes = {
        "good": 0,
        "defective": 1,
    }

    for class_name, label in classes.items():

        class_dir = split_dir / class_name

        for image_path in class_dir.glob("*.png"):

            image = cv2.imread(str(image_path))

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

            image = cv2.resize(
                image,
                (128, 128),
            )

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


print("Loading train...")
X_train, y_train = load_images("train")

print("Loading val...")
X_val, y_val = load_images("val")

print("Scaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

print("Training SVM...")

model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42,
)

model.fit(
    X_train,
    y_train,
)

predictions = model.predict(X_val)

accuracy = accuracy_score(
    y_val,
    predictions,
)

print("\nAccuracy:")
print(accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        predictions,
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_val,
        predictions,
    )
)

joblib.dump(
    model,
    MODEL_DIR / "hog_svm.pkl",
)

joblib.dump(
    scaler,
    MODEL_DIR / "hog_scaler.pkl",
)

print("\nModel saved.")
