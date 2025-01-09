from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Space

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
    def get(observation_type: int) -> type[_Observation]:
        match observation_type:
            case ObservationType.DICT:
                return DictObservation
            case ObservationType.FLATTENED:
                return FlattenedObservation
            case ObservationType.IMAGE:
                return ImageObservation
            case ObservationType.IMAGE_DICT:
                return ImageDictObservation
            case _:
                raise ValueError("Unknown observation type")


class _Observation(ABC):
    def __init__(self, warehouse: Warehouse):
        super().__init__()
        self.warehouse = warehouse
        self.msg_bits = self.warehouse.msg_bits
        self.sensor_range = self.warehouse.sensor_range
        self.normalised_coordinates = self.warehouse.normalised_coordinates
        self.n_agents = self.warehouse.n_agents
        self.grid_size = self.warehouse.grid_size

    @classmethod
    def from_warehouse(cls, warehouse: Warehouse):
        return cls(warehouse)

    @property
    @abstractmethod
    def space(self) -> Space:
        return NotImplementedError()


class DictObservation(_Observation):
    def __init__(self, warehouse: Warehouse):
        super().__init__(warehouse)
        self._obs_bits_for_self = 4 + len(Direction)
        self._obs_bits_per_agent = 1 + len(Direction) + self.msg_bits
        self._obs_bits_per_shelf = 2
        self._obs_bits_for_requests = 2

    @property
    @lru_cache
    def space(self):

        self._obs_sensor_locations = (1 + 2 * self.sensor_range) ** 2

        self._obs_length = (
            self._obs_bits_for_self
            + self._obs_sensor_locations * self._obs_bits_per_agent
            + self._obs_sensor_locations * self._obs_bits_per_shelf
        )

        max_grid_val = max(self.grid_size)
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


class FlattenedObservation(_Observation):
    def __init__(self, warehouse: Warehouse):
        super().__init__(warehouse)

    @property
    @lru_cache
    def space(self):
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


class ImageObservation(_Observation):
    def __init__(self, warehouse: Warehouse):
        super().__init__(warehouse)
        self.image_observation_layers = warehouse.image_observation_layers
        self.directional = warehouse.image_observation_directional

    @property
    @lru_cache
    def space(self):
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


class ImageDictObservation(_Observation):
    def __init__(self, warehouse: Warehouse):
        super().__init__(warehouse)

    @property
    @lru_cache
    def space(self):
        observation_space = ImageObservation(self.warehouse).space[0]

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
