from enum import Enum
from typing import TypeAlias
import numpy as np
import gymnasium as gym

from rware.utils.typing import Point, Direction


_LAYER_AGENTS = 0
_LAYER_SHELVES = 1

ID: TypeAlias = int


class AgentActionSpace:
    def __init__(self, msg_bits: int):
        super().__init__()
        self.msg_bits = msg_bits

    @property
    def space(self) -> gym.spaces.Discrete | gym.spaces.MultiDiscrete:
        if self.msg_bits <= 0:
            return gym.spaces.Discrete(len(AgentAction))
        else:
            return gym.spaces.MultiDiscrete([len(AgentAction), *self.msg_bits * (2,)])


class AgentAction(Enum):
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    TOGGLE_LOAD = 4


class Entity:
    def __init__(self, id_: ID, pos: Point):
        self.id = id_
        self.prev_pos = pos
        self.pos = pos

    @property
    def key(self) -> int:
        return self.id - 1

    @property
    def x(self) -> int:
        return self.pos.x

    @x.setter
    def x(self, value: int) -> None:
        self.pos = Point(value, self.pos.y)

    @property
    def y(self) -> int:
        return self.pos.y

    @y.setter
    def y(self, value: int) -> None:
        self.pos = Point(self.pos.x, value)


class Agent(Entity):
    def __init__(self, id_: ID, pos: Point, dir_: Direction, msg_bits: int):
        super().__init__(id_, pos)
        self.dir = dir_
        self.message = np.zeros(msg_bits)
        self.req_action: AgentAction | None = None
        self.carried_shelf: Shelf | None = None
        self.canceled_action = None
        self.has_delivered = False
        self.loaded = False

    @property
    def carried_shelf_exn(self):
        assert self.carried_shelf is not None, f"No shelf carried by agent {self.id}"
        return self.carried_shelf

    @property
    def collision_layers(self):
        if self.loaded:
            return (_LAYER_AGENTS, _LAYER_SHELVES)
        else:
            return (_LAYER_AGENTS,)

    def req_location(self, grid_size) -> Point:
        if self.req_action != AgentAction.FORWARD:
            return self.pos
        elif self.dir == Direction.UP:
            return Point(self.x, max(0, self.y - 1))
        elif self.dir == Direction.DOWN:
            return Point(self.x, min(grid_size[1] - 1, self.y + 1))
        elif self.dir == Direction.LEFT:
            return Point(max(0, self.x - 1), self.y)
        elif self.dir == Direction.RIGHT:
            return Point(min(grid_size[0] - 1, self.x + 1), self.y)

        raise ValueError(
            f"Direction is {self.dir}. Should be one of {[v for v in Direction]}"
        )

    def req_direction(self) -> Direction:
        wraplist = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        if self.req_action == AgentAction.RIGHT:
            return wraplist[(wraplist.index(self.dir) + 1) % len(wraplist)]
        elif self.req_action == AgentAction.LEFT:
            return wraplist[(wraplist.index(self.dir) - 1) % len(wraplist)]
        else:
            return self.dir


class Shelf(Entity):
    def __init__(self, id_: ID, pos: Point, color: int):
        super().__init__(id_, pos)
        self.is_requested = False
        self.color = color

    @property
    def collision_layers(self):
        return (_LAYER_SHELVES,)


class Goal(Entity):
    def __init__(self, id_: ID, pos: Point, color: int = 0):
        super().__init__(id_, pos)
        self.color = color
