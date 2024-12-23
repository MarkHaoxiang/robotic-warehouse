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

    SHELVES = 0  # binary layer indicating shelves (also indicates carried shelves)
    REQUESTS = 1  # binary layer indicating requested shelves
    AGENTS = 2  # binary layer indicating agents in the environment (no way to distinguish agents)
    AGENT_DIRECTION = 3  # layer indicating agent directions as int (see Direction enum + 1 for values)
    AGENT_LOAD = 4  # binary layer indicating agents with load
    GOALS = 5  # binary layer indicating goal/ delivery locations
    ACCESSIBLE = 6  # binary layer indicating accessible cells (all but occupied cells/ out of map)


class Point(NamedTuple):
    x: int
    y: int

    def normalise(self, grid_width: int, grid_height: int) -> tuple[float, float]:
        return (self.x / (grid_width - 1), self.y / (grid_height - 1))
