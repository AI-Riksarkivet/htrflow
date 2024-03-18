from collections import namedtuple
from typing import Sequence, TypeAlias

import numpy as np


Point = namedtuple("Point", ["x", "y"])
Bbox: TypeAlias = tuple[int, int, int, int]
Polygon: TypeAlias = Sequence[Point] | Sequence[tuple[int, int]]
Mask: TypeAlias = np.ndarray[np.uint8]
