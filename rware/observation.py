from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, TYPE_CHECKING

import networkx as nx
import numpy as np
import gymnasium.spaces as s
from gymnasium.spaces import Space

from rware.entity import Agent, Shelf, _LAYER_AGENTS, _LAYER_SHELVES
from rware.layout import Layout
from rware.utils.typing import Direction, ImageLayer, Point

if TYPE_CHECKING:
    # TODO markli: Really, we want to decouple this circular dependency by creating a WarehouseState class
    # and using dependency injection
    from .warehouse import Warehouse


class _VectorWriter:
    def __init__(self, size: int):
        self.vector = np.zeros(size, dtype=np.float32)
        self.idx = 0

    def write(self, data):
        data_size = len(data)
        self.vector[self.idx : self.idx + data_size] = data
        self.idx += data_size

    def skip(self, bits):
        self.idx += bits


class Observation(ABC):
    def __init__(self):
        super().__init__()

    def from_warehouse(self, warehouse: Warehouse):
        self.space = self.reset_space(warehouse)
        return self

    def reset_space(self, warehouse: Warehouse):
        self.msg_bits = warehouse.msg_bits
        self.sensor_range = warehouse.sensor_range
        self.n_agents = warehouse.n_agents
        space = self._reset_space(warehouse)
        return space

    @abstractmethod
    def _reset_space(self, warehouse: Warehouse) -> Space:
        raise NotImplementedError()

    def make_obs(self, warehouse: Warehouse) -> tuple[Any, ...]:
        self._prepare_obs(warehouse)
        return tuple(
            [self._make_agent_obs(agent, warehouse) for agent in warehouse.agents]
        )

    def _prepare_obs(self, warehouse: Warehouse):
        pass

    def _make_agent_obs(self, agent: Agent, warehouse: Warehouse):
        raise NotImplementedError()


class DictObservation(Observation):
    def __init__(self, normalised_coordinates: bool = False):
        super().__init__()
        self.normalised_coordinates = normalised_coordinates

    def _reset_space(self, warehouse):
        self._obs_bits_for_self = 4 + len(Direction)
        self._obs_bits_per_agent = 1 + len(Direction) + self.msg_bits
        self._obs_bits_per_shelf = 2
        self._obs_bits_for_requests = 2
        self._obs_sensor_locations = (1 + 2 * self.sensor_range) ** 2

        self._obs_length = (
            self._obs_bits_for_self
            + self._obs_sensor_locations * self._obs_bits_per_agent
            + self._obs_sensor_locations * self._obs_bits_per_shelf
        )

        max_grid_val = max(warehouse.grid_size)
        low = np.zeros(2)
        if self.normalised_coordinates:
            high = np.ones(2)
            dtype = np.float32
        else:
            high = np.ones(2) * max_grid_val
            dtype = np.int32
        location_space = s.Box(
            low=low,
            high=high,
            shape=(2,),
            dtype=dtype,
        )

        self_observation_dict_space = s.Dict(
            OrderedDict(
                {
                    "location": location_space,
                    "carrying_shelf": s.MultiBinary(1),
                    "direction": s.Discrete(4),
                    "on_highway": s.MultiBinary(1),
                }
            )
        )
        sensor_per_location_dict = OrderedDict(
            {
                "has_agent": s.MultiBinary(1),
                "direction": s.Discrete(4),
            }
        )
        if self.msg_bits > 0:
            sensor_per_location_dict["local_message"] = s.MultiBinary(self.msg_bits)
        sensor_per_location_dict.update(
            {
                "has_shelf": s.MultiBinary(1),
                "shelf_requested": s.MultiBinary(1),
            }
        )
        return _make_multiagent_space(
            s.Dict(
                OrderedDict(
                    {
                        "self": self_observation_dict_space,
                        "sensors": s.Tuple(
                            self._obs_sensor_locations
                            * (s.Dict(sensor_per_location_dict),)
                        ),
                    }
                )
            ),
            self.n_agents,
        )

    def _make_agent_obs(self, agent: Agent, warehouse: Warehouse):
        agents, shelves = make_local_obs(agent, warehouse, self.sensor_range)

        # write dictionary observations
        obs = {}
        pos = (
            agent.pos.normalise(*warehouse.grid_size)
            if self.normalised_coordinates
            else agent.pos
        )
        # --- self data
        obs["self"] = {
            "location": np.array(pos, dtype=np.int32),
            "carrying_shelf": [int(agent.carried_shelf is not None)],
            "direction": agent.dir.value,
            "on_highway": [int(warehouse.layout.is_highway(agent.pos))],
        }
        # --- sensor data
        obs["sensors"] = tuple({} for _ in range(self._obs_sensor_locations))

        # find neighboring agents
        for i, id_ in enumerate(agents):
            if id_ == 0:
                obs["sensors"][i]["has_agent"] = [0]
                obs["sensors"][i]["direction"] = 0
                obs["sensors"][i]["local_message"] = (
                    self.msg_bits * [0] if self.msg_bits > 0 else None
                )
            else:
                obs["sensors"][i]["has_agent"] = [1]
                obs["sensors"][i]["direction"] = warehouse.agents[id_ - 1].dir.value
                obs["sensors"][i]["local_message"] = (
                    warehouse.agents[id_ - 1].message if self.msg_bits > 0 else None
                )

        # find neighboring shelfs:
        for i, id_ in enumerate(shelves):
            if id_ == 0:
                obs["sensors"][i]["has_shelf"] = [0]
                obs["sensors"][i]["shelf_requested"] = [0]
            else:
                obs["sensors"][i]["has_shelf"] = [1]
                obs["sensors"][i]["shelf_requested"] = [
                    int(warehouse.shelves[id_ - 1] in warehouse.request_queue)
                ]

        return obs


class FlattenedObservation(Observation):
    def __init__(self, normalised_coordinates: bool = False):
        super().__init__()
        self.normalised_coordinates = normalised_coordinates

    def _reset_space(self, warehouse):
        observation_space = (
            DictObservation(self.normalised_coordinates).from_warehouse(warehouse).space
        )

        ma_spaces = []
        for sa_obs in observation_space:
            flatdim = s.flatdim(sa_obs)
            ma_spaces += [
                s.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        return s.Tuple(tuple(ma_spaces))

    def _make_agent_obs(self, agent: Agent, warehouse: Warehouse):
        agents, shelves = make_local_obs(agent, warehouse, self.sensor_range)

        # write flattened observations
        flatdim = s.flatdim(self.space[agent.id - 1])
        obs = _VectorWriter(flatdim)

        pos = (
            agent.pos.normalise(*warehouse.grid_size)
            if self.normalised_coordinates
            else agent.pos
        )

        obs.write([*pos, int(agent.carried_shelf is not None)])
        direction = np.zeros(4)
        direction[agent.dir.value] = 1.0
        obs.write(direction)
        obs.write([int(warehouse.layout.is_highway(agent.pos))])

        # 'has_agent': MultiBinary(1),
        # 'direction': Discrete(4),
        # 'local_message': MultiBinary(2)
        # 'has_shelf': MultiBinary(1),
        # 'shelf_requested': MultiBinary(1),

        for i, (id_agent, id_shelf) in enumerate(zip(agents, shelves)):
            if id_agent == 0:
                # no agent, direction, or message
                obs.write([0.0])  # no agent present
                obs.write([1.0, 0.0, 0.0, 0.0])  # agent direction
                obs.skip(self.msg_bits)  # agent message
            else:
                obs.write([1.0])  # agent present
                direction = np.zeros(4)
                direction[warehouse.agents[id_agent - 1].dir.value] = 1.0
                obs.write(direction)  # agent direction as onehot
                if self.msg_bits > 0:
                    obs.write(warehouse.agents[id_agent - 1].message)  # agent message
            if id_shelf == 0:
                obs.write([0.0, 0.0])  # no shelf or requested shelf
            else:
                obs.write(
                    [
                        1.0,
                        int(warehouse.shelves[id_shelf - 1] in warehouse.request_queue),
                    ]
                )  # shelf presence and request status
        return obs.vector


IMG_DTYPE = np.uint8


class ImageObservation(Observation):
    def __init__(self, directional: bool = True):
        super().__init__()
        self.directional = directional

    def _reset_space(self, warehouse: Warehouse):
        self.image_observation_layers = warehouse.default_global_image_layers
        obs_shape = (1 + 2 * self.sensor_range, 1 + 2 * self.sensor_range)

        layers_min = []
        layers_max = []
        for layer in self.image_observation_layers:
            layer_bounds = ImageLayer.get_bounds(layer, warehouse.layout.num_colors)
            for layer_min, layer_max in layer_bounds:
                layers_min.append(np.full(obs_shape, layer_min, dtype=IMG_DTYPE))
                layers_max.append(np.full(obs_shape, layer_max, dtype=IMG_DTYPE))

        # total observation
        min_obs = np.stack(layers_min)
        max_obs = np.stack(layers_max)
        return _make_multiagent_space(
            s.Box(min_obs, max_obs, dtype=IMG_DTYPE), self.n_agents
        )

    def _prepare_obs(self, warehouse):
        layers = make_global_image(
            warehouse,
            image_layers=self.image_observation_layers,
            padding_size=self.sensor_range,
        )
        self.global_layers = np.stack(layers)

    def _make_agent_obs(self, agent: Agent, warehouse: Warehouse):
        # global information was generated --> get information for agent
        start_x = agent.pos.x
        end_x = agent.pos.x + 2 * self.sensor_range + 1
        start_y = agent.pos.y
        end_y = agent.pos.y + 2 * self.sensor_range + 1
        obs = self.global_layers[:, start_x:end_x, start_y:end_y]

        if self.directional:
            obs = _rotate_obs(obs, agent.dir)
        return obs


class ShapedObservation(Observation):
    """When the global layout is known to all agents."""

    def __init__(
        self, image_observation_layers: list[ImageLayer], directional: bool = True
    ):
        super().__init__()
        self.image_observation_layers = image_observation_layers
        self.directional = directional

    def _make_layout_obs(self, layout: Layout) -> np.ndarray:
        shelves_locations = 1 - layout.highways.copy()
        goals = np.zeros_like(shelves_locations)
        for goal in layout.goals:
            goals[*goal.pos] = 1
        return np.stack((shelves_locations, goals), axis=0, dtype=np.int32)

    def _reset_space(self, warehouse: Warehouse):
        # Feature layers
        self.feature_space_dict = s.Dict(
            OrderedDict(
                {
                    "direction": s.Discrete(4),
                    "color": s.Discrete(warehouse.layout._num_colors + 1),
                    "carrying_shelf": s.MultiBinary(1),
                    "carried_shelf_color": s.Discrete(warehouse.layout._num_colors),
                    "on_highway": s.MultiBinary(1),
                    "on_shelf_requested": s.MultiBinary(1),
                    "on_shelf_color_correct": s.MultiBinary(1),
                    "closest_goal": s.Box(-1, 1, shape=(2,)),
                    "closest_uncollected_request": s.Box(-1, 1, shape=(2,)),
                    "closest_available_storage": s.Box(-1, 1, shape=(2,)),
                }
            )
        )
        feature_flat_dim = s.flatdim(self.feature_space_dict)
        feature_space = s.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(feature_flat_dim,),
            dtype=np.float32,
        )

        size = (1 + 2 * self.sensor_range, 1 + 2 * self.sensor_range)
        layers_min = []
        layers_max = []
        for layer in self.image_observation_layers:
            layer_bounds = ImageLayer.get_bounds(layer, warehouse.layout.num_colors)
            for layer_min, layer_max in layer_bounds:
                layers_min.append(np.full(size, layer_min, dtype=IMG_DTYPE))
                layers_max.append(np.full(size, layer_max, dtype=IMG_DTYPE))

        min_obs = np.stack(layers_min)
        max_obs = np.stack(layers_max)
        local_image_space = s.Box(min_obs, max_obs, dtype=IMG_DTYPE)

        agent_space = s.Dict(
            {
                "local_image": local_image_space,
                "features": feature_space,
                "position": s.Box(0, 1, (2,)),
            }
        )
        return _make_multiagent_space(agent_space, self.n_agents)

    def _prepare_obs(self, warehouse):
        self.world_image = np.stack(
            make_global_image(
                warehouse,
                image_layers=self.image_observation_layers,
            )
        )

    def _make_agent_obs(self, agent: Agent, warehouse: Warehouse):
        sensor_pad = [
            (0, 0),
            (warehouse.sensor_range, warehouse.sensor_range),
            (warehouse.sensor_range, warehouse.sensor_range),
        ]
        start_x = agent.pos.x
        end_x = agent.pos.x + 2 * self.sensor_range + 1
        start_y = agent.pos.y
        end_y = agent.pos.y + 2 * self.sensor_range + 1
        image = np.pad(self.world_image, sensor_pad, mode="constant")

        local_image = image[:, start_x:end_x, start_y:end_y]

        if self.directional:
            local_image = _rotate_obs(local_image, agent.dir)

        # Make feature obs
        feature_obs = _VectorWriter(self.space[agent.id - 1]["features"].shape[0])

        # Direction
        direction = np.zeros(4)
        direction[agent.dir.value] = 1.0
        feature_obs.write(direction)

        # Color of agent
        color = np.zeros(warehouse.layout._num_colors + 1)
        color[agent.color + 1] = 1.0
        feature_obs.write(color)

        # Carrying Shelf
        feature_obs.write(
            [
                int(agent.carried_shelf is not None),
            ]
        )

        # Color of carried shelf
        color = np.zeros(warehouse.layout._num_colors)
        if warehouse._has_shelf_at(agent.pos):
            color[warehouse._get_shelf_at_exn(agent.pos).color] = 1
        feature_obs.write(color)

        # On highway
        feature_obs.write([int(warehouse.layout.is_highway(agent.pos))])

        # On shelf requested / color correct
        is_requested = [0]
        is_color_correct = [0]
        if warehouse._has_shelf_at(agent.pos):
            if warehouse._get_shelf_at_exn(agent.pos).is_requested:
                is_requested[0] = 1
            if (
                warehouse._get_shelf_at_exn(agent.pos).color == agent.color
                or agent.color == -1
            ):
                is_color_correct[0] = 1
        feature_obs.write(is_requested)

        # Closest
        # Construct the traversability graph (approximate)
        if agent.carried_shelf:
            G = warehouse.layout.highway_traversal_graph

            # Agents can only travel on highways
            def distance_to(target: Point):
                try:
                    path = nx.shortest_path(G, agent.pos, target)
                    return len(path)
                except nx.NetworkXNoPath:
                    return len(G.nodes) + Point.manhattan_distance(agent.pos, target)
        else:
            # Agent can cross over shelves
            def distance_to(target: Point):
                return Point.manhattan_distance(agent.pos, target)

        def closest_position(positions: list[Point]):
            if len(positions) > 0:
                pos = min(positions, key=distance_to)
            else:
                pos = agent.pos
            return Point(pos.x - agent.pos.x, pos.y - agent.pos.y)

        if agent.carried_shelf:
            goal_positions = [
                goal.pos
                for goal in warehouse.layout.goals
                if goal.color == agent.carried_shelf.color
            ]
        else:
            goal_positions = []
        closest_goal = closest_position(goal_positions)

        carried_shelves = [a.carried_shelf for a in warehouse.agents]

        def filter_request(shelf: Shelf):
            assert shelf.is_requested
            if shelf in carried_shelves:
                return False
            if shelf.color != agent.color and agent.color != -1:
                return False
            return True

        uncollected_requests = [
            s.pos for s in warehouse.request_queue if filter_request(s)
        ]
        closest_uncollected_request = closest_position(uncollected_requests)

        available_storage = [
            Point(x, y)
            for x in range(warehouse.layout.grid_size[0])
            for y in range(warehouse.layout.grid_size[1])
        ]
        available_storage = [
            p
            for p in available_storage
            if not warehouse._has_shelf_at(p) and not warehouse.layout.is_highway(p)
        ]
        closest_available_storage = closest_position(available_storage)

        feature_obs.write(closest_goal.normalise(*warehouse.grid_size))
        feature_obs.write(closest_uncollected_request.normalise(*warehouse.grid_size))
        feature_obs.write(closest_available_storage.normalise(*warehouse.grid_size))

        pos_obs = np.array(agent.pos.normalise(*warehouse.grid_size), dtype=np.float32)

        return {
            "local_image": local_image,
            "features": feature_obs.vector,
            "position": pos_obs,
        }


class ImageDictObservation(Observation):
    def __init__(self, directional: bool = True):
        super().__init__()
        self.directional = directional

    def _reset_space(self, warehouse: Warehouse):
        self.image_generator = ImageObservation(
            directional=self.directional
        ).from_warehouse(warehouse)
        observation_space = self.image_generator.space[0]

        feature_space_dict = s.Dict(
            OrderedDict(
                {
                    "direction": s.Discrete(4),
                    "on_highway": s.MultiBinary(1),
                    "carrying_shelf": s.MultiBinary(1),
                }
            )
        )

        feature_flat_dim = s.flatdim(feature_space_dict)
        feature_space = s.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(feature_flat_dim,),
            dtype=np.float32,
        )

        return _make_multiagent_space(
            s.Dict({"image": observation_space, "features": feature_space}),
            self.n_agents,
        )

    def _prepare_obs(self, warehouse):
        return self.image_generator._prepare_obs(warehouse)

    def _make_agent_obs(self, agent, warehouse):
        image_obs = self.image_generator._make_agent_obs(agent, warehouse)
        feature_obs = _VectorWriter(self.space[agent.id - 1]["features"].shape[0])
        direction = np.zeros(4)
        direction[agent.dir.value] = 1.0
        feature_obs.write(direction)
        feature_obs.write(
            [
                int(warehouse.layout.is_highway(agent.pos)),
                int(agent.carried_shelf is not None),
            ]
        )
        return {
            "image": image_obs,
            "features": feature_obs.vector,
        }


def make_local_obs(agent: Agent, warehouse: Warehouse, sensor_range: int):
    min_x = agent.pos.x - sensor_range
    max_x = agent.pos.x + sensor_range + 1

    min_y = agent.pos.y - sensor_range
    max_y = agent.pos.y + sensor_range + 1

    # sensors
    if (
        (min_x < 0)
        or (min_y < 0)
        or (max_x > warehouse.grid_size[0])
        or (max_y > warehouse.grid_size[1])
    ):
        padded_agents = np.pad(
            warehouse.grid[_LAYER_AGENTS], sensor_range, mode="constant"
        )
        padded_shelves = np.pad(
            warehouse.grid[_LAYER_SHELVES], sensor_range, mode="constant"
        )
        # + self.sensor_range due to padding
        min_x += sensor_range
        max_x += sensor_range
        min_y += sensor_range
        max_y += sensor_range

    else:
        padded_agents = warehouse.grid[_LAYER_AGENTS]
        padded_shelves = warehouse.grid[_LAYER_SHELVES]

    agents = padded_agents[min_x:max_x, min_y:max_y].reshape(-1)
    shelves = padded_shelves[min_x:max_x, min_y:max_y].reshape(-1)
    return agents, shelves


def make_global_image(
    warehouse: Warehouse,
    image_layers: list[ImageLayer],
    padding_size: int | None = None,
):
    layers = []
    for layer_type in image_layers:
        layer = None
        match layer_type:
            case ImageLayer.SHELVES:
                layer = warehouse.grid[_LAYER_SHELVES].copy().astype(IMG_DTYPE)
                # set all occupied shelf cells to 1.0 (instead of shelf ID)
                layer[layer > 0.0] = 1.0
            case ImageLayer.REQUESTS:
                layer = np.zeros(warehouse.grid_size, dtype=IMG_DTYPE)
                for requested_shelf in warehouse.request_queue:
                    layer[*requested_shelf.pos] = 1.0
            case ImageLayer.AGENTS:
                layer = warehouse.grid[_LAYER_AGENTS].copy().astype(IMG_DTYPE)
                # set all occupied agent cells to 1.0 (instead of agent ID)
                layer[layer > 0] = 1
            case ImageLayer.AGENT_DIRECTION:
                layer = np.zeros(warehouse.grid_size, dtype=IMG_DTYPE)
                for ag in warehouse.agents:
                    agent_direction = ag.dir.value + 1
                    layer[*ag.pos] = agent_direction
            case ImageLayer.AGENT_COLOR:
                layer = np.zeros(warehouse.grid_size, dtype=IMG_DTYPE)
                for ag in warehouse.agents:
                    if ag.color != -1:
                        layer[*ag.pos] = ag.color + 1
                    else:
                        layer[*ag.pos] = -1
            case ImageLayer.AGENT_LOAD:
                layer = np.zeros(warehouse.grid_size, dtype=IMG_DTYPE)
                for ag in warehouse.agents:
                    if ag.carried_shelf is not None:
                        layer[*ag.pos] = 1
            case ImageLayer.GOALS:
                layer = np.zeros(warehouse.grid_size, dtype=IMG_DTYPE)
                for goal in warehouse.goals:
                    layer[goal.pos] = goal.color + 1
            case ImageLayer.GOALS_COLOR_ONE_HOT:
                concat_layers = np.zeros(
                    (warehouse.layout.num_colors, *warehouse.grid_size), dtype=IMG_DTYPE
                )
                for goal in warehouse.goals:
                    concat_layers[goal.color, *goal.pos] = 1
            case ImageLayer.ACCESSIBLE:
                layer = np.ones(warehouse.grid_size, dtype=IMG_DTYPE)
                for ag in warehouse.agents:
                    layer[*ag.pos] = 0
            case ImageLayer.STORAGE:
                concat_layers = warehouse.layout.storage.astype(IMG_DTYPE)
            case _:
                raise ValueError(f"Unknown image layer type: {layer_type}")
        if layer is not None:
            concat_layers = np.array([layer])
        # pad with 0s for out-of-map cells
        if padding_size:
            concat_layers = np.pad(
                concat_layers,
                ((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
                mode="constant",
            )  # type: ignore
        layers.append(concat_layers)
    return np.concat(layers, axis=0)


def _rotate_obs(obs: np.ndarray, dir: Direction):
    # rotate image to be in direction of agent
    # printing out obs matches the natural direction expected
    # where forward is the first row in the array
    if dir == Direction.UP:
        obs = np.rot90(obs, k=1, axes=(1, 2))
    elif dir == Direction.RIGHT:
        obs = np.rot90(obs, k=2, axes=(1, 2))
    elif dir == Direction.DOWN:
        obs = np.rot90(obs, k=3, axes=(1, 2))
    elif dir == Direction.LEFT:
        pass
    return obs


def _make_multiagent_space(agent_space: Space, n_agents: int) -> s.Tuple:
    return s.Tuple((agent_space for _ in range(n_agents)))


class ObservationRegistry:
    """A set of reasonable defaults for observations"""

    _DEFAULT_IMAGE_OBSERVATION_LAYERS = [
        ImageLayer.SHELVES,
        ImageLayer.REQUESTS,
        ImageLayer.AGENTS,
        ImageLayer.GOALS_COLOR_ONE_HOT,
        ImageLayer.STORAGE,
    ]

    DICT = DictObservation(normalised_coordinates=False)
    FLATTENED = FlattenedObservation(normalised_coordinates=False)
    IMAGE = ImageObservation(directional=True)
    IMAGE_DICT = ImageDictObservation(directional=True)
    SHAPED = ShapedObservation(
        image_observation_layers=_DEFAULT_IMAGE_OBSERVATION_LAYERS, directional=True
    )
