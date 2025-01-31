from __future__ import annotations

from enum import Enum
from typing import Any
import warnings

import gymnasium as gym
from gymnasium.utils import seeding
import networkx as nx
import numpy as np


from rware.utils.typing import ImageLayer, Direction, Point  # Re-export unused
from rware.utils import map_none
from rware.layout import Layout
from rware.entity import Action, Agent, Shelf, _LAYER_SHELVES, _LAYER_AGENTS
from rware.observation import Observation, ObservationRegistry, make_global_image


class RewardType(Enum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2
    SHAPED = 3


class ShapedReward:
    """Designed to make the environment easier to learn, while (probably) not impacting the optimal policy"""

    INVALID_OP_PENALTY = 0.01
    GOAL_POTENTIAL_REWARD = 0.05

    def __init__(self, layout: Layout):
        # TODO: Add a reward when not carrying a shelf?
        # We also add a potential based reward to encourage carrying shelves closer to (any) goal.
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
            ShapedReward.GOAL_POTENTIAL_REWARD
            - (self.distance_from_goal / maximum_distance)
            * ShapedReward.GOAL_POTENTIAL_REWARD
        )

        self.shelf_initial_potential = {}

    def get_reward_for_moving_target_shelf(
        self, agent: Agent, i_pos: Point, e_pos: Point
    ) -> float:
        shelf = agent.carried_shelf
        i_pot = self.potential_from_goal[*i_pos]
        e_pot = self.potential_from_goal[*e_pos]
        reward = e_pot - i_pot

        if shelf.id not in self.shelf_initial_potential:
            # Initial pickup
            self.shelf_initial_potential[shelf.id] = i_pot

        if self.distance_from_goal[*e_pos] == 0:
            # Reached goal
            reward += (
                0.5
                - ShapedReward.GOAL_POTENTIAL_REWARD
                + self.shelf_initial_potential[shelf.id]
            )
            self.shelf_initial_potential.pop(shelf.id)

        return reward


class Warehouse(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        shelf_columns: int | None = None,
        column_height: int | None = None,
        shelf_rows: int | None = None,
        n_agents: int = 3,
        msg_bits: int = 0,
        sensor_range: int = 1,
        request_queue_size: int = 5,
        max_inactivity_steps: int | None = 1000,
        max_steps: int | None = None,
        reward_type: RewardType = RewardType.GLOBAL,
        layout: Layout | str | None = None,
        observation_type: Observation = ObservationRegistry.FLATTENED,
        image_observation_layers: list[ImageLayer] = [
            ImageLayer.SHELVES,
            ImageLayer.REQUESTS,
            ImageLayer.AGENTS,
            ImageLayer.GOALS,
            ImageLayer.ACCESSIBLE,
        ],
        render_mode: str = "human",
    ):
        """The robotic warehouse environment

        Creates a grid world where multiple agents (robots)
        are supposed to collect shelfs, bring them to a goal
        and then return them.
        .. note:
            The grid looks like this:

            shelf
            columns
                vv
            ----------
            -XX-XX-XX-        ^
            -XX-XX-XX-  Column Height
            -XX-XX-XX-        v
            ----------
            -XX----XX-   <\
            -XX----XX-   <- Shelf Rows
            -XX----XX-   </
            ----------
            ----GG----

            G: is the goal positions where agents are rewarded if
            they bring the correct shelfs.

            The final grid size will be
            height: (column_height + 1) * shelf_rows + 2
            width: (2 + 1) * shelf_columns + 1

            The bottom-middle column will be removed to allow for
            robot queuing next to the goal locations

        :param shelf_columns: Number of columns in the warehouse
        :type shelf_columns: int
        :param column_height: Column height in the warehouse
        :type column_height: int
        :param shelf_rows: Number of columns in the warehouse
        :type shelf_rows: int
        :param n_agents: Number of spawned and controlled agents
        :type n_agents: int
        :param msg_bits: Number of communication bits for each agent
        :type msg_bits: int
        :param sensor_range: Range of each agents observation
        :type sensor_range: int
        :param request_queue_size: How many shelfs are simultaneously requested
        :type request_queue_size: int
        :param max_inactivity: Number of steps without a delivered shelf until environment finishes
        :type max_inactivity: Optional[int]
        :param reward_type: Specifies if agents are rewarded individually or globally
        :type reward_type: RewardType
        :param layout: A string for a custom warehouse layout. X are shelve locations, dots are corridors, and g are the goal locations. Ignores shelf_columns, shelf_height and shelf_rows when used.
        :type layout: str
        :param observation_type: Specifies type of observations
        :param image_observation_layers: Specifies types of layers observed if image-observations
            are used
        :type image_observation_layers: list[ImageLayer]
        """
        self.reward_range = (0, 1)
        self.reward_type = reward_type

        # Generate initial layout
        if layout is None:
            if shelf_columns is None or shelf_rows is None or column_height is None:
                raise ValueError("A layout is expected but none found.")
            self.layout = Layout.from_params(shelf_columns, shelf_rows, column_height)
        else:
            if isinstance(layout, str):
                self.layout = Layout.from_str(layout)
            else:
                self.layout = layout
        if self.layout._agents is not None:
            n = len(self.layout._agents)
            if n != n_agents:
                warnings.warn(
                    f"Provided layout agents {n} does not match n_agents {n_agents}."
                )
                n_agents = n

        self._update_layout(self.layout)

        self.n_agents = n_agents
        self.msg_bits = msg_bits
        self.sensor_range = sensor_range
        self.max_inactivity_steps = max_inactivity_steps

        self._cur_inactive_steps = None
        self._cur_steps = 0
        self.max_steps = max_steps

        # Compute action spaces
        sa_action_space = [len(Action), *msg_bits * (2,)]
        if len(sa_action_space) == 1:
            sa_action_space = gym.spaces.Discrete(sa_action_space[0])
        else:
            sa_action_space = gym.spaces.MultiDiscrete(sa_action_space)
        self.action_space = gym.spaces.Tuple(tuple(n_agents * [sa_action_space]))

        self.request_queue_size = request_queue_size
        self.request_queue: list[Shelf] = []

        self.agents: list[Agent] = []

        # Compute Observation spaces
        self.image_observation_layers = image_observation_layers
        self.obs_generator = observation_type.from_warehouse(self)
        self.observation_space = self.obs_generator.space

        self.global_image = None
        self.renderer = None
        self.render_mode = render_mode

    def _get_info(self):
        # TODO: Counter of delivered shelves?
        return {}

    def _recalc_grid(self):
        self.grid[:] = 0
        for s in self.shelves:
            self.grid[_LAYER_SHELVES, *s.pos] = s.id

        for a in self.agents:
            self.grid[_LAYER_AGENTS, *a.pos] = a.id

    def _update_layout(self, layout: Layout):
        layout.validate()
        self.layout = layout
        self.grid_size = self.layout.grid_size
        self.grid = self.layout.generate_grid()
        self.goals = self.layout.goals

        if self.reward_type == RewardType.SHAPED:
            self.shaped_reward = ShapedReward(layout)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            super().reset(seed=seed, options=options)

        self._cur_inactive_steps = 0
        self._cur_steps = 0

        # Update layout if provided
        if options is not None:
            new_layout = options.get("layout", self.layout)
            self._update_layout(new_layout)

        # Make shelves and agents
        self.shelves = self.layout.reset_shelves()
        self.agents = self.layout.reset_agents(
            self.msg_bits, (self.np_random, self.n_agents)
        )

        self._recalc_grid()

        self.request_queue = list(
            self.np_random.choice(
                self.shelves, size=self.request_queue_size, replace=False
            )
        )
        for shelf in self.request_queue:
            shelf.is_requested = True

        return self.obs_generator.make_obs(self), self._get_info()

    def step(
        self, actions: list[Action]
    ) -> tuple[list[np.ndarray], list[float], bool, bool, dict]:
        assert len(actions) == len(self.agents)

        # Mapping from agent id to rewards
        rewards = np.zeros(self.n_agents)

        for agent, action in zip(self.agents, actions):
            if self.msg_bits > 0:
                agent.req_action = Action(action[0])
                agent.message[:] = action[1:]
            else:
                agent.req_action = Action(action)

        commited_agents = set()

        G = nx.DiGraph()

        for agent in self.agents:
            start = agent.pos
            target = agent.req_location(self.grid_size)

            if (
                agent.carried_shelf
                and start != target
                and self._has_shelf_at(target)
                and not map_none(
                    self._get_agent_at(target), lambda a: a.carried_shelf
                )  # The shelf at the target is actually carried by another agent
            ):
                # there's a standing shelf at the target location
                # our agent is carrying a shelf so there's no way
                # this movement can succeed. Cancel it.
                agent.req_action = Action.NOOP
                G.add_edge(start, start)
                if self.reward_type == RewardType.SHAPED:
                    # Trying to put down a shelf when on a highway
                    rewards[agent.key] -= ShapedReward.INVALID_OP_PENALTY
            else:
                G.add_edge(start, target)

        wcomps = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]

        for comp in wcomps:
            try:
                # if we find a cycle in this component we have to
                # commit all nodes in that cycle, and nothing else
                cycle = nx.algorithms.find_cycle(comp)
                if len(cycle) == 2:
                    # we have a situation like this: [A] <-> [B]
                    # which is physically impossible. so skip
                    continue
                for edge in cycle:
                    start_node = edge[0]
                    agent_id = self.grid[_LAYER_AGENTS, start_node[0], start_node[1]]
                    if agent_id > 0:
                        commited_agents.add(agent_id)
            except nx.NetworkXNoCycle:
                longest_path = nx.algorithms.dag_longest_path(comp)
                for x, y in longest_path:
                    agent_id = self.grid[_LAYER_AGENTS, x, y]
                    if agent_id:
                        commited_agents.add(agent_id)

        commited_agents = set([self.agents[id_ - 1] for id_ in commited_agents])
        failed_agents = set(self.agents) - commited_agents

        for agent in failed_agents:
            assert agent.req_action == Action.FORWARD, "Other actions can't fail."
            agent.req_action = Action.NOOP

        for agent in self.agents:
            agent.prev_pos = agent.pos

            if agent.req_action == Action.FORWARD:
                # Move forward
                agent.pos = agent.req_location(self.grid_size)
                if agent.carried_shelf:
                    # Move a shelf
                    agent.carried_shelf.pos = agent.pos
                    if (
                        agent.carried_shelf in self.request_queue
                        and self.reward_type == RewardType.SHAPED
                    ):
                        rewards[agent.key] += (
                            self.shaped_reward.get_reward_for_moving_target_shelf(
                                agent, agent.prev_pos, agent.pos
                            )
                        )

            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                # Rotate direction
                agent.dir = agent.req_direction()
            elif (
                agent.req_action == Action.TOGGLE_LOAD
                and not agent.carried_shelf
                and self._has_shelf_at(agent.pos)
            ):
                # Pick up a shelf
                agent.carried_shelf = self._get_shelf_at(agent.pos)
                if (
                    agent.carried_shelf.is_requested
                    and self.reward_type == RewardType.SHAPED
                ):
                    rewards[agent.key] += 0.25
            elif agent.req_action == Action.TOGGLE_LOAD and agent.carried_shelf:
                # Try to put down a shelf
                if not self.layout.is_highway(agent.pos):
                    if agent.has_delivered:
                        if self.reward_type == RewardType.TWO_STAGE:
                            rewards[agent.key] += 0.5
                        elif self.reward_type == RewardType.SHAPED:
                            rewards[agent.key] += 0.25
                    elif (
                        agent.carried_shelf.is_requested
                        and self.reward_type == RewardType.SHAPED
                    ):
                        rewards[agent.key] -= 0.25

                    agent.carried_shelf = None
                    agent.has_delivered = False
                elif self.reward_type == RewardType.SHAPED:
                    # Trying to put down a shelf when on a highway
                    rewards[agent.key] -= ShapedReward.INVALID_OP_PENALTY
            elif (
                agent.req_action == Action.TOGGLE_LOAD
                and not agent.carried_shelf
                and not self._has_shelf_at(agent.pos)
                and self.reward_type == RewardType.SHAPED
            ):
                rewards[agent.key] -= ShapedReward.INVALID_OP_PENALTY

        self._recalc_grid()

        shelf_delivered = False
        for goal in self.goals:
            shelf = self._get_shelf_at(goal)

            if shelf is None or not shelf.is_requested:
                continue
            # a shelf was successfully delived.
            shelf_delivered = True
            # remove from queue and replace it
            candidates = [s for s in self.shelves if s not in self.request_queue]
            new_request: Shelf = self.np_random.choice(candidates)
            self.request_queue[self.request_queue.index(shelf)] = new_request
            shelf.is_requested = False
            new_request.is_requested = True
            # also reward the agents
            agent = self._get_agent_at(goal)
            assert agent is not None
            agent.has_delivered = True
            match self.reward_type:
                case RewardType.GLOBAL:
                    rewards += 1
                case RewardType.INDIVIDUAL:
                    rewards[agent.key] += 1
                case RewardType.TWO_STAGE:
                    rewards[agent.key] += 0.5
                case RewardType.SHAPED:
                    # This is done in get_reward_for_moving_shelf
                    pass
                case _:
                    pass

        if shelf_delivered:
            self._cur_inactive_steps = 0
        else:
            self._cur_inactive_steps += 1
        self._cur_steps += 1

        if (
            self.max_inactivity_steps
            and self._cur_inactive_steps >= self.max_inactivity_steps
        ) or (self.max_steps and self._cur_steps >= self.max_steps):
            done = True
        else:
            done = False
        truncated = False

        # print(rewards)
        new_obs = self.obs_generator.make_obs(self)
        info = self._get_info()
        return new_obs, list(rewards), done, truncated, info

    def render(self):
        if not self.renderer:
            from rware.rendering import Viewer

            self.renderer = Viewer(self.grid_size)

        return self.renderer.render(
            self, return_rgb_array=self.render_mode == "rgb_array"
        )

    def close(self):
        if self.renderer:
            self.renderer.close()

    def seed(self, seed=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

    def _get_shelf_at(self, pos: Point) -> Shelf | None:
        shelf_id = self.grid[_LAYER_SHELVES, *pos]
        return self.shelves[shelf_id - 1] if shelf_id else None

    def _has_shelf_at(self, pos: Point) -> bool:
        return self._get_shelf_at(pos) is not None

    def _get_agent_at(self, pos: Point) -> Agent | None:
        agent_id = self.grid[_LAYER_AGENTS, *pos]
        return self.agents[agent_id - 1] if agent_id else None

    def _has_agent_at(self, pos: Point) -> bool:
        return self._get_agent_at(pos) is not None

    def get_global_image(
        self,
        image_layers=[
            ImageLayer.SHELVES,
            ImageLayer.GOALS,
        ],
        recompute=False,
        pad_to_shape=None,
    ) -> np.ndarray:
        """
        Get global image observation
        :param image_layers: image layers to include in global image
        :param recompute: bool whether image should be recomputed or taken from last computation
            (for default params, image will be constant for environment so no recomputation needed
             but if agent or request information is included, then should be recomputed)
         :param pad_to_shape: if given than pad environment global image shape into this
             shape (if doesn't fit throw exception)
        """
        if recompute or self.global_image is None:
            layers = make_global_image(self, image_layers, None)
            self.global_image = np.stack(layers)
            if pad_to_shape is not None:
                padding_dims = [
                    pad_dim - global_dim
                    for pad_dim, global_dim in zip(
                        pad_to_shape, self.global_image.shape
                    )
                ]
                assert all([dim >= 0 for dim in padding_dims])
                pad_before = [pad_dim // 2 for pad_dim in padding_dims]
                pad_after = [
                    pad_dim // 2 if pad_dim % 2 == 0 else pad_dim // 2 + 1
                    for pad_dim in padding_dims
                ]
                self.global_image = np.pad(
                    self.global_image,
                    pad_width=tuple(zip(pad_before, pad_after)),
                    mode="constant",
                    constant_values=0,
                )
        return self.global_image
