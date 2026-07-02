# Industrial Defect Detection

Industrial Defect Detection is a computer vision project for binary quality-control classification on the MVTec AD `bottle` category. The system detects whether an input bottle image is `good` or `defective` and includes both a classical machine learning baseline and a deep learning model.

The project demonstrates a complete computer vision pipeline: data acquisition, preprocessing, handcrafted feature extraction, CNN-based representation learning, model evaluation, Grad-CAM explainability, and FastAPI inference.

## Project Scope

The selected real-world scenario is industrial quality control. In manufacturing environments, visual inspection is used to reduce defective products reaching customers. This project models that workflow by classifying product images into two classes:

- `good`: non-defective bottle samples.
- `defective`: bottle samples with visible anomalies.

The implementation uses the `bottle` category from the MVTec Anomaly Detection dataset.

## Repository Structure

```text
backend/
  app/
    main.py                 FastAPI application
    inference.py            Upload-to-prediction adapter
computer_vision/
  dataset.py                MVTec sample collection utilities
  split_dataset.py          Binary train/validation/test split
  train_hog_svm.py          HOG feature extraction + SVM training
  evaluate_hog.py           HOG + SVM test evaluation
  train_resnet.py           ResNet50 fine-tuning
  evaluate_resnet.py        ResNet50 test evaluation
  gradcam_resnet.py         Grad-CAM visualization
  predict.py                Single-image inference
outputs/
  hog_confusion_matrix.png
  resnet_confusion_matrix.png
  gradcam_overlay.png
models/
  hog_svm.pkl
  hog_scaler.pkl
  resnet50_bottle.pth
```

The `dataset/`, `processed_dataset/`, `models/`, and `outputs/` directories are ignored by Git because they may contain large or generated artifacts. They should be regenerated locally when needed.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

Download the MVTecAD dataset (https://www.mvtec.com/research-teaching/datasets/mvtec-ad/downloads or direct https://www.mydrive.ch/shares/150452/132a93367fb17cdf968dfb5c4013f6e7/download/420937370-1629958698/bottle.tar.xz) and place the extracted data under:

```text
dataset/
  bottle/
    train/
    test/
    ground_truth/
```

The project currently targets the `bottle` category. The loader also preserves defect type metadata and mask paths when available.

Create the binary classification split:

```bash
python computer_vision/split_dataset.py
```

## Pipeline

1. Data acquisition and preprocessing
   - Collect MVTec `bottle` images.
   - Convert the original anomaly-detection layout into a binary classification dataset.
   - Resize images for the target model.
   - Normalize tensors with ImageNet statistics for ResNet50.
   - Apply augmentation during ResNet50 training.

2. Feature representation
   - Classical path: grayscale conversion, resizing to `128x128`, HOG descriptors.
   - Deep learning path: ImageNet-pretrained ResNet50 backbone with a custom classification head.

3. Core logic
   - HOG features are classified with an RBF-kernel SVM.
   - ResNet50 is fine-tuned by freezing the backbone and training a custom head.

4. Post-processing and explanation
   - Softmax probabilities are converted into a final class label and confidence score.
   - Grad-CAM produces a heatmap overlay to show which image regions influence the CNN prediction.

## Training

Train the classical baseline:

```bash
python computer_vision/train_hog_svm.py
```

Train the ResNet50 model:

```bash
python computer_vision/train_resnet.py
```

The ResNet50 script automatically selects `mps`, `cuda`, or `cpu` depending on local hardware availability.

## Evaluation

Evaluate HOG + SVM:

```bash
python computer_vision/evaluate_hog.py
```

Evaluate ResNet50:

```bash
python computer_vision/evaluate_resnet.py
```

Current test results on the local binary split:

| Model                    | Accuracy | Defective Precision | Defective Recall | Defective F1 |
| ------------------------ | -------: | ------------------: | ---------------: | -----------: |
| HOG + SVM                |   95.45% |             100.00% |           77.78% |       87.50% |
| ResNet50 fine-tuned head |   93.18% |             100.00% |           66.67% |       80.00% |

The confusion matrics are generated as:

- `outputs/hog_confusion_matrix.png`
- `outputs/resnet_confusion_matrix.png`

## Grad-CAM

Generate a GradCAM overlay for a defective test image:

```bash
python computer_vision/gradcam_resnet.py
```

The output is saved to:

```text
outputs/gradcam_overlay.png
```

Grad-CAM is used as an interpretability step to inspect whether the ResNet50 model focuses on visually relevant regions.

## API

Start the API locally:

```bash
uvicorn backend.app.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Run prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@path/to/image.png"
```

Example response:

```json
{
  "status": true,
  "data": {
    "image": "/tmp/tmp_image.png",
    "prediction": "defective",
    "confidence": 0.92,
    "model": "resnet50"
  }
}
```

## Docker

Build and run with Docker Compose:

```bash
docker compose up --build
```

The API is exposed on:

```text
http://127.0.0.1:8000
```

## Security and Limitations

- The current API is intended for local demonstration.
- CORS is open in the current FastAPI configuration and should be restricted before deployment removing the "*" wildcard.
- The model is trained on a limited dataset category and should not be treated as a universal industrial inspection system.
- False negatives are the highest-risk failure mode because defective products may be classified as good.