from collections import namedtuple
from typing import Sequence, TypeAlias


Point = namedtuple("Point", ["x", "y"])
Bbox: TypeAlias = tuple[int, int, int, int]
Polygon: TypeAlias = Sequence[Point] | Sequence[tuple[int, int]]
