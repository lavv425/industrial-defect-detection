import pytest

from dataset import collect_mvtec_samples


def build_mvtec_tree(root):
    """Creates a minimal MVTec-style directory layout under `root`.

    collect_mvtec_samples only globs paths (never opens the files),
    so empty placeholder files are enough to exercise the logic.
    """
    category = root / "bottle"

    train_good = category / "train" / "good"
    train_good.mkdir(parents=True)
    (train_good / "000.png").touch()
    (train_good / "001.png").touch()

    test_good = category / "test" / "good"
    test_good.mkdir(parents=True)
    (test_good / "000.png").touch()

    test_broken = category / "test" / "broken"
    test_broken.mkdir(parents=True)
    (test_broken / "000.png").touch()  # will have a matching mask
    (test_broken / "001.png").touch()  # will NOT have a mask

    ground_truth = category / "ground_truth" / "broken"
    ground_truth.mkdir(parents=True)
    (ground_truth / "000_mask.png").touch()

    return category


def test_collects_all_samples(tmp_path):
    build_mvtec_tree(tmp_path)
    samples = collect_mvtec_samples(tmp_path, "bottle")
    # 2 train/good + 1 test/good + 2 test/broken
    assert len(samples) == 5


def test_labels_and_names_are_consistent(tmp_path):
    build_mvtec_tree(tmp_path)
    samples = collect_mvtec_samples(tmp_path, "bottle")

    good = [s for s in samples if s.defect_type == "good"]
    broken = [s for s in samples if s.defect_type == "broken"]

    assert len(good) == 3
    assert len(broken) == 2
    assert all(s.label == 0 and s.label_name == "good" for s in good)
    assert all(s.label == 1 and s.label_name == "defective" for s in broken)


def test_mask_path_resolves_only_when_file_exists(tmp_path):
    build_mvtec_tree(tmp_path)
    samples = collect_mvtec_samples(tmp_path, "bottle")

    broken = {s.path.name: s for s in samples if s.defect_type == "broken"}

    assert broken["000.png"].mask_path is not None
    assert broken["000.png"].mask_path.name == "000_mask.png"
    assert broken["001.png"].mask_path is None


def test_good_samples_never_have_masks(tmp_path):
    build_mvtec_tree(tmp_path)
    samples = collect_mvtec_samples(tmp_path, "bottle")
    assert all(s.mask_path is None for s in samples if s.label == 0)


def test_missing_category_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_mvtec_samples(tmp_path, "does_not_exist")
