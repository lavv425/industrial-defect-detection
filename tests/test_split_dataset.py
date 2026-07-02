from pathlib import Path
from types import SimpleNamespace

import pytest

from split_dataset import get_source_split


def sample_with_path(path: str) -> SimpleNamespace:
    # get_source_split only inspects sample.path.parts
    return SimpleNamespace(path=Path(path))


@pytest.mark.parametrize(
    "path, expected",
    [
        ("dataset/bottle/train/good/000.png", "original_train"),
        ("dataset/bottle/test/broken/001.png", "original_test"),
        ("dataset/bottle/other/000.png", "unknown"),
    ],
)
def test_get_source_split(path, expected):
    assert get_source_split(sample_with_path(path)) == expected


def test_train_takes_precedence_when_both_present():
    sample = sample_with_path("dataset/train/test/000.png")
    assert get_source_split(sample) == "original_train"
