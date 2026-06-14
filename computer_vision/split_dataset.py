from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

from dataset import collect_mvtec_samples

OUTPUT_DIR = Path.cwd() / "processed_dataset" / "bottle_binary"


def get_source_split(sample) -> str:
    parts = sample.path.parts

    if "train" in parts:
        return "original_train"

    if "test" in parts:
        return "original_test"

    return "unknown"


def copy_samples(samples, split_name: str) -> None:
    for index, sample in enumerate(samples):
        label_dir = "good" if sample.label == 0 else "defective"

        target_dir = OUTPUT_DIR / split_name / label_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        source_split = get_source_split(sample)

        target_filename = (
            f"{index:04d}_{source_split}_{sample.defect_type}_{sample.path.name}"
        )

        target_path = target_dir / target_filename

        shutil.copyfile(sample.path, target_path)


def main() -> None:
    samples = collect_mvtec_samples("dataset", "bottle")

    labels = [sample.label for sample in samples]

    train_samples, temp_samples = train_test_split(
        samples,
        test_size=0.30,
        random_state=42,
        stratify=labels,
    )

    temp_labels = [sample.label for sample in temp_samples]

    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=0.50,
        random_state=42,
        stratify=temp_labels,
    )

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    copy_samples(train_samples, "train")
    copy_samples(val_samples, "val")
    copy_samples(test_samples, "test")

    print("Split completed")
    print(f"Train: {len(train_samples)}")
    print(f"Val:   {len(val_samples)}")
    print(f"Test:  {len(test_samples)}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
