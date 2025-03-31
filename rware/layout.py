from __future__ import annotations
from functools import cached_property

import numpy as np
import networkx as nx

from rware.entity import Agent, Shelf, Goal, Color
from rware.utils.typing import Point, Direction, ImageLayer

_COLLISION_LAYERS = 2


class Layout:
    def __init__(
        self,
        grid_size: tuple[int, int],
        goals: list[Goal],
        storage: np.ndarray,
        num_colors: int = 1,
        agents: list[tuple[Point, Direction, Color]] | None = None,
    ):
        """A layout determines the initial setting of the warehouse.

        Args:
            grid_size (tuple[int, int]): Shape of the map.
            goals (list[Position]): List of goals
            highways (np.ndarray): Boolean map indicator of positions that are NOT occupied by an intial shelf
            agents (list[Point, Direction] | None): A list of initial agent positions and directions. If None provided, then randomly generated.
        """
        super().__init__()
        self.grid_size = grid_size
        self.goals = goals

        self._num_colors = num_colors
        if len(storage.shape) == 2:
            storage = storage[None, :, :]
        self._storage = storage
        self._shelf_colors = np.argmax(storage, axis=0)

        self._highways: np.ndarray = 1 - storage.sum(axis=0)

        self._agents = agents

        self.validate()

    def validate(self):
        assert len(self.goals) >= 1, "At least one goal position must be provided."
        assert self._storage.shape[0] == self._num_colors, "Number of colors mismatch."
        assert self._storage.sum(axis=0).max() == 1, (
            "Multiple shelves at the same position."
        )

        color_has_goal = [False for _ in range(self._num_colors)]
        for goal in self.goals:
            color_has_goal[goal.color] = True
        assert all(color_has_goal), "Not all colors have a goal."

        assert self._highways.shape[0] == self.grid_size[0], (
            "Implicit grid_size:width from highways does not match grid_size."
        )
        assert self._highways.shape[1] == self.grid_size[1], (
            "Implicit grid_size:height from highways does not match grid_size."
        )
        for goal in self.goals:
            assert self.is_highway(goal.pos), (
                f"A shelf cannot be placed onto a goal allocated position. Position {goal.pos}"
            )

    def is_highway(self, pos: Point) -> bool:
        return self.highways[*pos]

    def get_color_exn(self, pos: Point) -> int:
        if self.is_highway(pos):
            raise ValueError(f"{pos} is a highway")
        return self._shelf_colors[*pos]

    def reset_shelves(self) -> list[Shelf]:
        shelf_counter = 0
        shelves: list[Shelf] = []
        for x, y in zip(
            np.indices(self.grid_size)[0].reshape(-1),
            np.indices(self.grid_size)[1].reshape(-1),
        ):
            pos = Point(x, y)
            if not self.is_highway(pos):
                shelf_counter += 1
                shelves.append(Shelf(shelf_counter, pos, self.get_color_exn(pos)))
        return shelves

    def reset_agents(
        self, msg_bits: int, params: tuple[np.random.Generator, int] | None = None
    ):
        if self._agents is None:
            assert params is not None, (
                "If agent positions are randomly generated, extra parameter are required."
            )
            rng, n_agents = params
            agent_locs = rng.choice(
                np.arange(self.grid_size[0] * self.grid_size[1]),
                size=n_agents,
                replace=False,
            )
            agent_locs = np.unravel_index(agent_locs, self.grid_size)  # type: ignore
            # And direction
            agent_dirs = rng.choice(Direction, size=n_agents)  # type: ignore
            agents = [
                Agent(
                    i + 1, Point(x, y), dir_, msg_bits, -1
                )  # IDs start from 1 since 0 represents nothing in observation
                for i, (x, y, dir_) in enumerate(zip(*agent_locs, agent_dirs))
            ]
        else:
            # TODO markli: Validate
            agents = [
                Agent(i + 1, Point(*pos), dir_, msg_bits, color)
                for i, (pos, dir_, color) in enumerate(self._agents)
            ]
        return agents

    def generate_grid(self) -> np.ndarray:
        """Generates an empty grid"""
        return np.zeros((_COLLISION_LAYERS, *self.grid_size), dtype=np.int32)

    @cached_property
    def highway_traversal_graph(self) -> nx.Graph:
        G = nx.Graph()
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                p = Point(x, y)
                G.add_node(p)
                if self.is_highway(p):
                    # Add edges in the four cardinal directions
                    if x > 0:
                        G.add_edge(p, Point(x - 1, y))
                    if x < self.grid_size[0] - 1:
                        G.add_edge(p, Point(x + 1, y))
                    if y > 0:
                        G.add_edge(p, Point(x, y - 1))
                    if y < self.grid_size[1] - 1:
                        G.add_edge(p, Point(x, y + 1))
        return G

    @property
    def highways(self) -> np.ndarray:
        return self._highways.copy()

    @property
    def storage(self) -> np.ndarray:
        return self._storage.copy()

    @property
    def num_colors(self) -> int:
        return self._num_colors

    @staticmethod
    def from_params(shelf_columns: int, shelf_rows: int, column_height: int) -> Layout:
        assert shelf_columns % 2 == 1, "Only odd number of shelf columns is supported"
        grid_size = (
            (2 + 1) * shelf_columns + 1,  # Width
            (column_height + 1) * shelf_rows + 2,  # Height
        )
        column_height = column_height
        goals = [
            Goal(0, Point(grid_size[0] // 2 - 1, grid_size[1] - 1)),
            Goal(1, Point(grid_size[0] // 2, grid_size[1] - 1)),
        ]

        shelves = np.zeros(grid_size, dtype=np.uint8)

        def highway_func(x, y):
            is_on_vertical_highway = x % 3 == 0
            is_on_horizontal_highway = y % (column_height + 1) == 0
            is_on_delivery_row = y == grid_size[1] - 1
            is_on_queue = (y > grid_size[1] - (column_height + 3)) and (
                x == grid_size[0] // 2 - 1 or x == grid_size[0] // 2
            )
            return (
                is_on_vertical_highway
                or is_on_horizontal_highway
                or is_on_delivery_row
                or is_on_queue
            )

        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                shelves[x, y] = 1 - int(highway_func(x, y))

        return Layout(grid_size, goals, shelves)

    @staticmethod
    def from_str(layout: str) -> Layout:
        layout = layout.strip().replace(" ", "")
        grid_height = layout.count("\n") + 1
        lines = layout.split("\n")
        grid_width = len(lines[0])
        for line in lines:
            assert len(line) == grid_width, "Layout must be rectangular"

        goals: list[Goal] = []
        grid_size = (grid_width, grid_height)
        shelves = np.zeros(grid_size, dtype=np.uint8)

        goal_count = 0
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                assert char.lower() in "gx."
                if char.lower() == "g":
                    goals.append(Goal(id_=goal_count, pos=Point(x, y)))
                    goal_count += 1
                elif char.lower() == "x":
                    shelves[x, y] = 1

        return Layout(grid_size=grid_size, goals=goals, storage=shelves, num_colors=1)

    @staticmethod
    def from_image(
        image: np.ndarray,
        image_layers: list[ImageLayer] = [ImageLayer.STORAGE, ImageLayer.GOALS],
    ) -> Layout:
        """The partial inverse of Warehouse.get_global_image

        Args:
            image (np.ndarray): An image of shape (C, W, H).
            layers (list: ImageLayer): Describes the channels used to generate image. C - len(layers). Highways is assumed to be the inverse of shelves. If AGENT is present but no AGENT_DIRECTION is given, then all directions are assumed pointing up.


        Returns:
            Layout: Layout that would generate image.
        """
        _, w, h = image.shape

        # Initial grid size
        grid_size = (w, h)

        # Shelves
        assert any(layer == ImageLayer.STORAGE for layer in image_layers), (
            f"Requires at least one STORAGE channel within image, but received {image_layers}"
        )
        shelf_layers = [
            image[i]
            for i, layer in enumerate(image_layers)
            if layer == ImageLayer.STORAGE
        ]
        shelves = np.stack(shelf_layers)
        num_colors = shelves.shape[0]

        # Goals
        assert ImageLayer.GOALS in image_layers, (
            f"Requires GOALS as a channel within image, but recieved {image_layers}"
        )
        goal_layer = image[image_layers.index(ImageLayer.GOALS)]
        goal_layer = goal_layer.astype(np.int32)
        goals = []
        for i, pos in enumerate(np.argwhere(goal_layer)):
            point = Point(pos[0], pos[1])  # type: ignore
            goals.append(Goal(i, point, color=goal_layer[*point] - 1))

        # Agents
        agents = None
        if ImageLayer.AGENTS in image_layers:
            agent_layer = image[image_layers.index(ImageLayer.AGENTS)]
            positions = np.argwhere(agent_layer)
            agent_positions = [Point(*pos.tolist()) for pos in positions]

            # Get agent directions if provided
            if ImageLayer.AGENT_DIRECTION in image_layers:
                directions = image[image_layers.index(ImageLayer.AGENT_DIRECTION)]
            else:
                # All pointing up
                directions = np.ones((w, h), dtype=np.int32)

            # Get agent colors if provided
            if ImageLayer.AGENT_COLOR in image_layers:
                colors = image[image_layers.index(ImageLayer.AGENT_COLOR)] - 1
            else:
                colors = np.full((w, h), -1, dtype=np.int32)
            colors = np.maximum(colors, -1)

            agents = [
                (pos, Direction(directions[*pos] - 1), int(colors[*pos]))
                for pos in agent_positions
            ]
        return Layout(
            grid_size=grid_size,
            goals=goals,
            storage=shelves,
            agents=agents,
            num_colors=num_colors,
        )

    @property
    def n_goals(self):
        return len(self.goals)
