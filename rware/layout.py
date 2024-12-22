from __future__ import annotations

import numpy as np

from rware.entity import Shelf

_COLLISION_LAYERS = 2


class Layout:
    """A layout determines the initial setting of the warehouse."""

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid: np.ndarray,
        goals: list[tuple[int, int]],
        highways: np.ndarray,
    ):
        super().__init__()
        self.grid_size = grid_size
        self._grid = grid
        self.goals = goals
        self._highways = highways  # Boolean indicator

    def validate(self):
        assert len(self.goals) >= 1, "At least one goal must be provided."
        assert (
            self._grid.shape[1],
            self._grid.shape[2],
        ) == self.grid_size, "Grid size does not match grid given"

    def is_highway(self, x, y) -> bool:
        return self.highways[y, x]

    def reset_shelves(self) -> list[Shelf]:
        shelf_counter = 0
        shelves: list[Shelf] = []
        for y, x in zip(
            np.indices(self.grid_size)[0].reshape(-1),
            np.indices(self.grid_size)[1].reshape(-1),
        ):
            if not self.is_highway(x, y):
                shelf_counter += 1
                shelves.append(Shelf(shelf_counter, x, y))
        return shelves

    @property
    def grid(self) -> np.ndarray:
        return self._grid.copy()

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
        grid = np.zeros((_COLLISION_LAYERS, *grid_size), dtype=np.int32)
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

        return Layout(grid_size, grid, goals, highways)

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
        grid = np.zeros((_COLLISION_LAYERS, *grid_size), dtype=np.int32)
        highways = np.zeros(grid_size, dtype=np.uint8)

        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                assert char.lower() in "gx."
                if char.lower() == "g":
                    goals.append((x, y))
                    highways[y, x] = 1
                elif char.lower() == ".":
                    highways[y, x] = 1

        return Layout(grid_size, grid, goals, highways)
