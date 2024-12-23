from __future__ import annotations

import numpy as np

from rware.entity import Shelf
from rware.utils.typing import Point

_COLLISION_LAYERS = 2


class Layout:
    def __init__(
        self,
        grid_size: tuple[int, int],
        goals: list[Point],
        highways: np.ndarray,
    ):
        """A layout determines the initial setting of the warehouse.

        Args:
            grid_size (tuple[int, int]): Shape of the map.
            goals (list[Position]): List of goals
            highways (np.ndarray): Boolean map indicator of positions that are NOT occupied by an intial shelf
        """
        super().__init__()
        self.grid_size = grid_size
        self.goals = goals
        self._highways = highways

    def validate(self):
        assert len(self.goals) >= 1, "At least one goal position must be provided."

    def is_highway(self, pos: Point) -> bool:
        return self.highways[pos.y, pos.x]

    def reset_shelves(self) -> list[Shelf]:
        shelf_counter = 0
        shelves: list[Shelf] = []
        for y, x in zip(
            np.indices(self.grid_size)[0].reshape(-1),
            np.indices(self.grid_size)[1].reshape(-1),
        ):
            if not self.is_highway(Point(x, y)):
                shelf_counter += 1
                shelves.append(Shelf(shelf_counter, Point(x, y)))
        return shelves

    def generate_grid(self) -> np.ndarray:
        """Generates an empty grid"""
        return np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)

    @property
    def highways(self) -> np.ndarray:
        return self._highways.copy()

    @staticmethod
    def from_params(shelf_columns: int, shelf_rows: int, column_height: int) -> Layout:
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"
        grid_size = (
            (column_height + 1) * shelf_rows + 2,
            (2 + 1) * shelf_columns + 1,
        )
        column_height = column_height
        goals = [
            (grid_size[1] // 2 - 1, grid_size[0] - 1),
            (grid_size[1] // 2, grid_size[0] - 1),
        ]

        highways = np.zeros(grid_size, dtype=np.uint8)

        def highway_func(x, y):
            is_on_vertical_highway = x % 3 == 0
            is_on_horizontal_highway = y % (column_height + 1) == 0
            is_on_delivery_row = y == grid_size[0] - 1
            is_on_queue = (y > grid_size[0] - (column_height + 3)) and (
                x == grid_size[1] // 2 - 1 or x == grid_size[1] // 2
            )
            return (
                is_on_vertical_highway
                or is_on_horizontal_highway
                or is_on_delivery_row
                or is_on_queue
            )

        for x in range(grid_size[1]):
            for y in range(grid_size[0]):
                highways[y, x] = int(highway_func(x, y))

        return Layout(grid_size, goals, highways)

    @staticmethod
    def from_str(layout: str):
        layout = layout.strip().replace(" ", "")
        grid_height = layout.count("\n") + 1
        lines = layout.split("\n")
        grid_width = len(lines[0])
        for line in lines:
            assert len(line) == grid_width, "Layout must be rectangular"

        goals = []
        grid_size = (grid_height, grid_width)
        highways = np.zeros(grid_size, dtype=np.uint8)

        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                assert char.lower() in "gx."
                if char.lower() == "g":
                    goals.append((x, y))
                    highways[y, x] = 1
                elif char.lower() == ".":
                    highways[y, x] = 1

        return Layout(grid_size, goals, highways)
