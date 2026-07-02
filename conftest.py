import sys
from pathlib import Path

# repo root and computer_vision directory importable in tests
ROOT_DIR = Path(__file__).resolve().parent
COMPUTER_VISION_DIR = ROOT_DIR / "computer_vision"

for path in (ROOT_DIR, COMPUTER_VISION_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
