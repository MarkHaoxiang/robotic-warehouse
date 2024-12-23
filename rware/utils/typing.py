from typing import NamedTuple


class Point(NamedTuple):
    x: int
    y: int

    def normalise(self, grid_width: int, grid_height: int) -> tuple[float, float]:
        return (self.x / (grid_width - 1), self.y / (grid_height - 1))
