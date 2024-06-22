from pathlib import Path
import sys


PROJ_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = PROJ_ROOT / "external"
XMEM_ROOT = PROJ_ROOT / "external" / "XMem"

# 11 colors following the Tab10 colormap from matplotlib, refer to
# https://matplotlib.org/stable/users/explain/colors/colormaps.html
SEG_CLASS_COLORS = (
    (0, 0, 0),  # background
    (31, 119, 180),  # Color 1
    (255, 127, 14),  # Color 2
    (44, 160, 44),  # Color 3
    (214, 39, 40),  # Color 4
    (148, 103, 189),  # Color 5
    (140, 86, 75),  # Color 6
    (227, 119, 194),  # Color 7
    (127, 127, 127),  # Color 8
    (188, 189, 34),  # Color 9
    (23, 190, 207),  # Color 10
)


def add_path(path):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
