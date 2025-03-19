from __future__ import annotations

from typing import NamedTuple
from enum import Enum


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class ImageLayer(Enum):
    """
    Input layers of image-style observations
    """

    # binary layer indicating shelves (also indicates carried shelves)
    SHELVES = 0
    # binary layer indicating requested shelves
    REQUESTS = 1
    # binary layer indicating agents in the environment (no way to distinguish agents)
    AGENTS = 2
    # layer indicating agent directions as int (see Direction enum + 1 for values)
    AGENT_DIRECTION = 3
    # binary layer indicating agents with load
    AGENT_LOAD = 4
    # colors of each image
    AGENT_COLOR = 5
    # layer indicating goal/ delivery locations as int (color + 1 for values)
    GOALS = 6
    # one-hot layer indicating goal/ delivery locations. Layer for each color.
    GOALS_COLOR_ONE_HOT = 7
    # binary layer indicating accessible cells (all but occupied cells/ out of map)
    ACCESSIBLE = 8
    # binary layer indicating locations where shelves can be placed (non-highway)
    STORAGE = 9

    @staticmethod
    def get_bounds(layer: ImageLayer, num_colors: int) -> list[tuple[int, int]]:
        match layer:
            case ImageLayer.AGENT_DIRECTION:
                return [(0, len(Direction))]
            case ImageLayer.GOALS | ImageLayer.AGENT_COLOR:
                return [(0, num_colors + 1)]
            # One hot
            case ImageLayer.GOALS_COLOR_ONE_HOT | ImageLayer.STORAGE:
                return [(0, 1) for _ in range(num_colors)]
            case _:
                return [(0, 1)]


class Point(NamedTuple):
    x: int
    y: int

    def normalise(self, grid_width: int, grid_height: int) -> tuple[float, float]:
        return (self.x / (grid_width - 1), self.y / (grid_height - 1))

    @staticmethod
    def manhattan_distance(a: Point, b: Point) -> int:
        return abs(a.x - b.x) + abs(a.y - b.y)
