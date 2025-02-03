from abc import ABC
from dataclasses import dataclass

import networkx as nx
import numpy as np

from rware.utils.typing import Point
from rware.layout import Layout
from rware.entity import ID


@dataclass
class DeliveredRequest:
    agent_id: ID
    shelf_id: ID
    goal: Point


@dataclass
class PickupShelf:
    agent_id: ID
    shelf_id: ID
    loc: Point
    is_requested: bool


@dataclass
class DropoffShelf:
    agent_id: ID
    shelf_id: ID
    loc: Point
    is_delivered: bool
    is_requested: bool


@dataclass
class MoveShelf:
    agent_id: ID
    shelf_id: ID
    start: Point
    end: Point
    is_requested: bool


# Immutable descriptions of things that have happened
Event = DeliveredRequest | PickupShelf | DropoffShelf | MoveShelf


class Reward(ABC):
    def reset(self, n_agents: int, layout: Layout):
        self._n_agents = n_agents
        self._layout = layout

    def compute_reward(self, events: list[Event]) -> list[float]:
        rewards = np.zeros(self._n_agents)
        for event in events:
            self.process_event(event, rewards)

        return list(rewards)

    def process_event(self, event: Event, rewards: np.ndarray) -> None:
        pass


class Global(Reward):
    def process_event(self, event: Event, rewards: np.ndarray):
        if isinstance(event, DeliveredRequest):
            rewards += 1


class Individual(Reward):
    def process_event(self, event: Event, rewards: np.ndarray):
        if isinstance(event, DeliveredRequest):
            rewards[event.agent_id - 1] += 1


class TwoStage(Reward):
    def process_event(self, event: Event, rewards: np.ndarray):
        if isinstance(event, DeliveredRequest):
            rewards[event.agent_id - 1] += 0.5
        elif isinstance(event, DropoffShelf) and event.is_delivered:
            rewards[event.agent_id - 1] += 0.5


class Shaped(Reward):
    """Designed to make the environment easier to learn, while (probably) not impacting the optimal policy"""

    def __init__(self, goal_potential_reward: float = 0.05):
        super().__init__()
        self.GOAL_POTENTIAL_REWARD = goal_potential_reward

    def reset(self, n_agents, layout):
        super().reset(n_agents, layout)
        self.distance_from_goal = np.ones_like(layout.highways) * np.inf

        G = nx.Graph()

        for x in range(layout.grid_size[0]):
            for y in range(layout.grid_size[1]):
                p = Point(x, y)
                G.add_node(p)
                if layout.is_highway(p):
                    # Add edges in the four cardinal directions
                    if x > 0:
                        G.add_edge(p, Point(x - 1, y))
                    if x < layout.grid_size[0] - 1:
                        G.add_edge(p, Point(x + 1, y))
                    if y > 0:
                        G.add_edge(p, Point(x, y - 1))
                    if y < layout.grid_size[1] - 1:
                        G.add_edge(p, Point(x, y + 1))

        for goal in layout.goals:
            paths = nx.single_source_shortest_path(G=G, source=goal)
            for target, path in paths.items():
                self.distance_from_goal[*target] = min(
                    self.distance_from_goal[*target], len(path) - 1
                )

        maximum_distance = np.max(
            self.distance_from_goal[np.isfinite(self.distance_from_goal)]
        )
        self.distance_from_goal = np.minimum(self.distance_from_goal, maximum_distance)
        self.potential_from_goal = (
            self.GOAL_POTENTIAL_REWARD
            - (self.distance_from_goal / maximum_distance) * self.GOAL_POTENTIAL_REWARD
        )

        self.shelf_initial_potential: dict[int, float] = {}

    def process_event(self, event, rewards):
        if isinstance(event, DeliveredRequest):
            reward = (
                0.5
                - self.GOAL_POTENTIAL_REWARD
                + self.shelf_initial_potential.get(event.shelf_id)
            )
            self.shelf_initial_potential.pop(event.shelf_id)
            rewards[event.agent_id - 1] += reward
        elif isinstance(event, DropoffShelf):
            if event.is_delivered:
                rewards[event.agent_id - 1] += 0.25
            elif event.is_requested:
                rewards[event.agent_id - 1] -= 0.25
        elif isinstance(event, PickupShelf) and event.is_requested:
            rewards[event.agent_id - 1] += 0.25
            self.shelf_initial_potential[event.shelf_id] = self.potential_from_goal[
                *event.loc
            ]
        elif isinstance(event, MoveShelf) and event.is_requested:
            i_pot = self.potential_from_goal[*event.start]
            e_pot = self.potential_from_goal[*event.end]
            reward = e_pot - i_pot
            rewards[event.agent_id - 1] += reward


class RewardRegistry:
    GLOBAL = Global()
    INDIVIDUAL = Individual()
    TWO_STAGE = TwoStage()
    SHAPED = Shaped(goal_potential_reward=0.05)
