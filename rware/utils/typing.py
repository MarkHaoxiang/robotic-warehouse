from typing import NamedTuple
from enum import Enum


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Point(NamedTuple):
    x: int
    y: int

    def normalise(self, grid_width: int, grid_height: int) -> tuple[float, float]:
        return (self.x / (grid_width - 1), self.y / (grid_height - 1))
