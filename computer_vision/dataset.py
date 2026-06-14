from pathlib import Path
from dataclasses import dataclass
from typing import Literal

LabelName = Literal["good", "defective"]


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int
    label_name: LabelName
    defect_type: str
    mask_path: Path | None


def collect_mvtec_samples(dataset_root: str | Path, category: str = "bottle") -> list[ImageSample]:
    root = Path(dataset_root) / category

    if not root.exists():
        raise FileNotFoundError(f"Dataset category not found: {root}")

    samples: list[ImageSample] = []

    train_good_dir = root / "train" / "good"
    for img_path in sorted(train_good_dir.glob("*.png")):
        samples.append(
            ImageSample(
                path=img_path,
                label=0,
                label_name="good",
                defect_type="good",
                mask_path=None,
            )
        )

    test_dir = root / "test"
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue

        defect_type = defect_dir.name
        is_good = defect_type == "good"

        for img_path in sorted(defect_dir.glob("*.png")):
            mask_path = None

            if not is_good:
                mask_path = (
                    root
                    / "ground_truth"
                    / defect_type
                    / f"{img_path.stem}_mask.png"
                )

                if not mask_path.exists():
                    mask_path = None

            samples.append(
                ImageSample(
                    path=img_path,
                    label=0 if is_good else 1,
                    label_name="good" if is_good else "defective",
                    defect_type=defect_type,
                    mask_path=mask_path,
                )
            )

    return samples


def print_dataset_summary(samples: list[ImageSample]) -> None:
    total = len(samples)
    good = sum(sample.label == 0 for sample in samples)
    defective = sum(sample.label == 1 for sample in samples)

    print(f"Total images: {total}")
    print(f"Good images: {good}")
    print(f"Defective images: {defective}")

    defect_types: dict[str, int] = {}

    for sample in samples:
        defect_types[sample.defect_type] = defect_types.get(sample.defect_type, 0) + 1

    print("\nImages by defect type:")
    for defect_type, count in sorted(defect_types.items()):
        print(f"- {defect_type}: {count}")


if __name__ == "__main__":
    samples = collect_mvtec_samples("dataset", "bottle")
    print_dataset_summary(samples)