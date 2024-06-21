from pathlib import Path
import sys


def add_path(path):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# Add PROJ_ROOT to sys.path
PROJ_ROOT = Path(__file__).resolve().parents[1]

add_path(PROJ_ROOT)
