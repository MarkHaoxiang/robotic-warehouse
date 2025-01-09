from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Space

from rware.entity import Agent, _LAYER_AGENTS, _LAYER_SHELVES
from rware.utils.typing import Direction, ImageLayer

if TYPE_CHECKING:
    # TODO markli: Really, we want to decouple this circular dependency by creating a WarehouseState class
    # and using dependency injection
    from warehouse import Warehouse


class ObservationType(Enum):
    DICT = 0
    FLATTENED = 1
    IMAGE = 2
    IMAGE_DICT = 3

    @staticmethod
    def get(observation_type: ObservationType) -> type[_Observation]:
        types = [
            DictObservation,
            FlattenedObservation,
            ImageObservation,
            ImageDictObservation,
        ]
        return types[observation_type.value]


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


class _Observation(ABC):
    def __init__(self, warehouse: Warehouse):
        super().__init__()
        self.warehouse = warehouse
        self.msg_bits = self.warehouse.msg_bits
        self.sensor_range = self.warehouse.sensor_range
        self.normalised_coordinates = self.warehouse.normalised_coordinates
        self.n_agents = self.warehouse.n_agents
        warehouse.grid_size = self.warehouse.grid_size

        self.space = self._reset_space(warehouse)

    @classmethod
    def from_warehouse(cls, warehouse: Warehouse):
        return cls(warehouse)

    @abstractmethod
    def _reset_space(self, warehouse: Warehouse) -> Space:
        return NotImplementedError()

    @abstractmethod
    def make_obs(self, agent: Agent, warehouse: Warehouse):
        return NotImplementedError()


class DictObservation(_Observation):
    def __init__(self, warehouse: Warehouse):
        super().__init__(warehouse)

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
        location_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(2,),
            dtype=dtype,
        )

        self_observation_dict_space = gym.spaces.Dict(
            OrderedDict(
                {
                    "location": location_space,
                    "carrying_shelf": gym.spaces.MultiBinary(1),
                    "direction": gym.spaces.Discrete(4),
                    "on_highway": gym.spaces.MultiBinary(1),
                }
            )
        )
        sensor_per_location_dict = OrderedDict(
            {
                "has_agent": gym.spaces.MultiBinary(1),
                "direction": gym.spaces.Discrete(4),
            }
        )
        if self.msg_bits > 0:
            sensor_per_location_dict["local_message"] = gym.spaces.MultiBinary(
                self.msg_bits
            )
        sensor_per_location_dict.update(
            {
                "has_shelf": gym.spaces.MultiBinary(1),
                "shelf_requested": gym.spaces.MultiBinary(1),
            }
        )
        return gym.spaces.Tuple(
            tuple(
                [
                    gym.spaces.Dict(
                        OrderedDict(
                            {
                                "self": self_observation_dict_space,
                                "sensors": gym.spaces.Tuple(
                                    self._obs_sensor_locations
                                    * (gym.spaces.Dict(sensor_per_location_dict),)
                                ),
                            }
                        )
                    )
                    for _ in range(self.n_agents)
                ]
            )
        )

    def make_obs(self, agent: Agent, warehouse: Warehouse):
        min_x = agent.pos.x - self.sensor_range
        max_x = agent.pos.x + self.sensor_range + 1

        min_y = agent.pos.y - self.sensor_range
        max_y = agent.pos.y + self.sensor_range + 1

        # sensors
        if (
            (min_x < 0)
            or (min_y < 0)
            or (max_x > warehouse.grid_size[0])
            or (max_y > warehouse.grid_size[1])
        ):
            padded_agents = np.pad(
                warehouse.grid[_LAYER_AGENTS], self.sensor_range, mode="constant"
            )
            padded_shelfs = np.pad(
                warehouse.grid[_LAYER_SHELVES], self.sensor_range, mode="constant"
            )
            # + self.sensor_range due to padding
            min_x += self.sensor_range
            max_x += self.sensor_range
            min_y += self.sensor_range
            max_y += self.sensor_range

        else:
            padded_agents = warehouse.grid[_LAYER_AGENTS]
            padded_shelfs = warehouse.grid[_LAYER_SHELVES]

        agents = padded_agents[min_x:max_x, min_y:max_y].reshape(-1)
        shelves = padded_shelfs[min_x:max_x, min_y:max_y].reshape(-1)

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
            "carrying_shelf": [int(agent.carrying_shelf is not None)],
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


class FlattenedObservation(_Observation):
    def __init__(self, warehouse: Warehouse):
        super().__init__(warehouse)

    def _reset_space(self, warehouse):
        observation_space = DictObservation(self.warehouse).space

        ma_spaces = []
        for sa_obs in observation_space:
            flatdim = gym.spaces.flatdim(sa_obs)
            ma_spaces += [
                gym.spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        return gym.spaces.Tuple(tuple(ma_spaces))

    def make_obs(self, agent: Agent, warehouse: Warehouse):
        min_x = agent.pos.x - self.sensor_range
        max_x = agent.pos.x + self.sensor_range + 1

        min_y = agent.pos.y - self.sensor_range
        max_y = agent.pos.y + self.sensor_range + 1

        # sensors
        if (
            (min_x < 0)
            or (min_y < 0)
            or (max_x > warehouse.grid_size[0])
            or (max_y > warehouse.grid_size[1])
        ):
            padded_agents = np.pad(
                warehouse.grid[_LAYER_AGENTS], self.sensor_range, mode="constant"
            )
            padded_shelfs = np.pad(
                warehouse.grid[_LAYER_SHELVES], self.sensor_range, mode="constant"
            )
            # + self.sensor_range due to padding
            min_x += self.sensor_range
            max_x += self.sensor_range
            min_y += self.sensor_range
            max_y += self.sensor_range

        else:
            padded_agents = warehouse.grid[_LAYER_AGENTS]
            padded_shelfs = warehouse.grid[_LAYER_SHELVES]

        agents = padded_agents[min_x:max_x, min_y:max_y].reshape(-1)
        shelves = padded_shelfs[min_x:max_x, min_y:max_y].reshape(-1)

        # write flattened observations
        flatdim = gym.spaces.flatdim(self.space[agent.id - 1])
        obs = _VectorWriter(flatdim)

        pos = (
            agent.pos.normalise(*warehouse.grid_size)
            if self.normalised_coordinates
            else agent.pos
        )

        obs.write([*pos, int(agent.carrying_shelf is not None)])
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


class ImageObservation(_Observation):
    def __init__(self, warehouse: Warehouse):
        super().__init__(warehouse)

    def _reset_space(self, warehouse):
        self.image_observation_layers = warehouse.image_observation_layers
        self.directional = warehouse.image_observation_directional
        observation_shape = (1 + 2 * self.sensor_range, 1 + 2 * self.sensor_range)

        layers_min = []
        layers_max = []
        for layer in self.image_observation_layers:
            if layer == ImageLayer.AGENT_DIRECTION:
                # directions as int
                layer_min = np.zeros(observation_shape, dtype=np.float32)
                layer_max = np.ones(observation_shape, dtype=np.float32) * max(
                    [d.value + 1 for d in Direction]
                )
            else:
                # binary layer
                layer_min = np.zeros(observation_shape, dtype=np.float32)
                layer_max = np.ones(observation_shape, dtype=np.float32)
            layers_min.append(layer_min)
            layers_max.append(layer_max)

        # total observation
        min_obs = np.stack(layers_min)
        max_obs = np.stack(layers_max)
        return gym.spaces.Tuple(
            tuple([gym.spaces.Box(min_obs, max_obs, dtype=np.float32)] * self.n_agents)
        )

    def make_obs(self, agent, warehouse):
        if agent.id == 1:
            layers = []

            # first agent's observation --> update global observation layers
            for layer_type in warehouse.image_observation_layers:

                match layer_type:
                    case ImageLayer.SHELVES:
                        layer = warehouse.grid[_LAYER_SHELVES].copy().astype(np.float32)
                        # set all occupied shelf cells to 1.0 (instead of shelf ID)
                        layer[layer > 0.0] = 1.0
                    case ImageLayer.REQUESTS:
                        layer = np.zeros(warehouse.grid_size, dtype=np.float32)
                        for requested_shelf in warehouse.request_queue:
                            layer[*requested_shelf.pos] = 1.0
                    case ImageLayer.AGENTS:
                        layer = warehouse.grid[_LAYER_AGENTS].copy().astype(np.float32)
                        # set all occupied agent cells to 1.0 (instead of agent ID)
                        layer[layer > 0.0] = 1.0
                    case ImageLayer.AGENT_DIRECTION:
                        layer = np.zeros(warehouse.grid_size, dtype=np.float32)
                        for ag in warehouse.agents:
                            agent_direction = ag.dir.value + 1
                            layer[*ag.pos] = float(agent_direction)
                    case ImageLayer.AGENT_LOAD:
                        layer = np.zeros(warehouse.grid_size, dtype=np.float32)
                        for ag in warehouse.agents:
                            if ag.carrying_shelf is not None:
                                layer[*ag.pos] = 1.0
                    case ImageLayer.GOALS:
                        layer = np.zeros(warehouse.grid_size, dtype=np.float32)
                        for goal in warehouse.goals:
                            layer[*goal] = 1.0
                    case ImageLayer.ACCESSIBLE:
                        layer = np.ones(warehouse.grid_size, dtype=np.float32)
                        for ag in warehouse.agents:
                            layer[*ag.pos] = 0.0
                    case _:
                        raise ValueError(f"Unknown image layer type: {layer_type}")

                # pad with 0s for out-of-map cells
                layer = np.pad(layer, self.sensor_range, mode="constant")
                layers.append(layer)
            self.global_layers = np.stack(layers)

        # global information was generated --> get information for agent
        start_x = agent.pos.x
        end_x = agent.pos.x + 2 * self.sensor_range + 1
        start_y = agent.pos.y
        end_y = agent.pos.y + 2 * self.sensor_range + 1
        obs = self.global_layers[:, start_x:end_x, start_y:end_y]

        if warehouse.image_observation_directional:
            # rotate image to be in direction of agent
            if agent.dir == Direction.DOWN:
                # rotate by 180 degrees (clockwise)
                obs = np.rot90(obs, k=2, axes=(1, 2))
            elif agent.dir == Direction.LEFT:
                # rotate by 90 degrees (clockwise)
                obs = np.rot90(obs, k=3, axes=(1, 2))
            elif agent.dir == Direction.RIGHT:
                # rotate by 270 degrees (clockwise)
                obs = np.rot90(obs, k=1, axes=(1, 2))
            # no rotation needed for UP direction
        return obs


class ImageDictObservation(_Observation):
    def __init__(self, warehouse: Warehouse):
        super().__init__(warehouse)

    def _reset_space(self, warehouse):
        self.image_generator = ImageObservation(warehouse)
        observation_space = self.image_generator.space[0]

        feature_space = gym.spaces.Dict(
            OrderedDict(
                {
                    "direction": gym.spaces.Discrete(4),
                    "on_highway": gym.spaces.MultiBinary(1),
                    "carrying_shelf": gym.spaces.MultiBinary(1),
                }
            )
        )

        feature_flat_dim = gym.spaces.flatdim(feature_space)
        feature_space = gym.spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(feature_flat_dim,),
            dtype=np.float32,
        )

        return gym.spaces.Tuple(
            tuple(
                [
                    gym.spaces.Dict(
                        {"image": observation_space, "features": feature_space}
                    )
                    for _ in range(self.n_agents)
                ]
            )
        )

    def make_obs(self, agent, warehouse):
        image_obs = self.image_generator.make_obs(agent, warehouse)
        feature_obs = _VectorWriter(self.space[agent.id - 1]["features"].shape[0])
        direction = np.zeros(4)
        direction[agent.dir.value] = 1.0
        feature_obs.write(direction)
        feature_obs.write(
            [
                int(warehouse.layout.is_highway(agent.pos)),
                int(agent.carrying_shelf is not None),
            ]
        )
        return {
            "image": image_obs,
            "features": feature_obs.vector,
        }
