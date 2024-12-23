from __future__ import annotations

from collections import OrderedDict
from enum import Enum
from typing import Any

import gymnasium as gym
from gymnasium.utils import seeding
import networkx as nx
import numpy as np


from rware.utils.typing import Direction, ImageLayer
from rware.layout import Layout
from rware.entity import Action, Agent, Shelf, _LAYER_SHELVES, _LAYER_AGENTS


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


class RewardType(Enum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2


class ObservationType(Enum):
    DICT = 0
    FLATTENED = 1
    IMAGE = 2
    IMAGE_DICT = 3


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
        observation_type: ObservationType = ObservationType.FLATTENED,
        image_observation_layers: list[ImageLayer] = [
            ImageLayer.SHELVES,
            ImageLayer.REQUESTS,
            ImageLayer.AGENTS,
            ImageLayer.GOALS,
            ImageLayer.ACCESSIBLE,
        ],
        image_observation_directional: bool = True,
        normalised_coordinates: bool = False,
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
        :param image_observation_directional: Specifies whether image observations should be
            rotated to be directional (agent perspective) if image-observations are used
        :type image_observation_directional: bool
        :param normalised_coordinates: Specifies whether absolute coordinates should be normalised
            with respect to total warehouse size
        :type normalised_coordinates: bool
        """

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
        self._update_layout(self.layout)

        self.n_agents = n_agents
        self.msg_bits = msg_bits
        self.sensor_range = sensor_range
        self.max_inactivity_steps = max_inactivity_steps
        self.reward_type = reward_type
        self.reward_range = (0, 1)

        self._cur_inactive_steps = None
        self._cur_steps = 0
        self.max_steps = max_steps

        self.normalised_coordinates = normalised_coordinates

        sa_action_space = [len(Action), *msg_bits * (2,)]
        if len(sa_action_space) == 1:
            sa_action_space = gym.spaces.Discrete(sa_action_space[0])
        else:
            sa_action_space = gym.spaces.MultiDiscrete(sa_action_space)
        self.action_space = gym.spaces.Tuple(tuple(n_agents * [sa_action_space]))

        self.request_queue_size = request_queue_size
        self.request_queue: list[Shelf] = []

        self.agents: list[Agent] = []

        # default values:
        self.fast_obs = None
        self.image_obs = None
        self.image_dict_obs = None
        if observation_type == ObservationType.IMAGE:
            self.observation_space = self._use_image_obs(
                image_observation_layers, image_observation_directional
            )
        elif observation_type == ObservationType.IMAGE_DICT:
            self.observation_space = self._use_image_dict_obs(
                image_observation_layers, image_observation_directional
            )

        else:
            # used for DICT observation type and needed as preceeding stype to generate
            # FLATTENED observations as well
            self.observation_space = self._use_slow_obs()

            # for performance reasons we
            # can flatten the obs vector
            if observation_type == ObservationType.FLATTENED:
                self.observation_space = self._use_fast_obs()

        self.global_image = None
        self.renderer = None
        self.render_mode = render_mode

    def _use_image_obs(self, image_observation_layers, directional=True):
        """
        Set image observation space
        :param image_observation_layers (list[ImageLayer]): list of layers to use as image channels
        :param directional (bool): flag whether observations should be directional (pointing in
            direction of agent or north-wise)
        """
        self.image_obs = True
        self.fast_obs = False
        self.image_dict_obs = True
        self.image_observation_directional = directional
        self.image_observation_layers = image_observation_layers

        observation_shape = (1 + 2 * self.sensor_range, 1 + 2 * self.sensor_range)

        layers_min = []
        layers_max = []
        for layer in image_observation_layers:
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

    def _use_image_dict_obs(self, image_observation_layers, directional=True):
        """
        Get image dictionary observation with image and flattened feature vector
        :param image_observation_layers (list[ImageLayer]): list of layers to use as image channels
        :param directional (bool): flag whether observations should be directional (pointing in
            direction of agent or north-wise)
        """
        image_obs_space = self._use_image_obs(image_observation_layers, directional)[0]
        self.image_obs = False
        self.image_dict_obs = True
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
                        {"image": image_obs_space, "features": feature_space}
                    )
                    for _ in range(self.n_agents)
                ]
            )
        )

    def _use_slow_obs(self):
        self.fast_obs = False

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

    def _use_fast_obs(self):
        if self.fast_obs:
            return self.observation_space

        self.fast_obs = True
        ma_spaces = []
        for sa_obs in self.observation_space:
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

    def _make_img_obs(self, agent: Agent):
        # write image observations
        if agent.id == 1:
            layers = []

            # first agent's observation --> update global observation layers
            for layer_type in self.image_observation_layers:

                match layer_type:
                    case ImageLayer.SHELVES:
                        layer = self.grid[_LAYER_SHELVES].copy().astype(np.float32)
                        # set all occupied shelf cells to 1.0 (instead of shelf ID)
                        layer[layer > 0.0] = 1.0
                    case ImageLayer.REQUESTS:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for requested_shelf in self.request_queue:
                            layer[*requested_shelf.pos] = 1.0
                    case ImageLayer.AGENTS:
                        layer = self.grid[_LAYER_AGENTS].copy().astype(np.float32)
                        # set all occupied agent cells to 1.0 (instead of agent ID)
                        layer[layer > 0.0] = 1.0
                    case ImageLayer.AGENT_DIRECTION:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for ag in self.agents:
                            agent_direction = ag.dir.value + 1
                            layer[*ag.pos] = float(agent_direction)
                    case ImageLayer.AGENT_LOAD:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for ag in self.agents:
                            if ag.carrying_shelf is not None:
                                layer[*ag.pos] = 1.0
                    case ImageLayer.GOALS:
                        layer = np.zeros(self.grid_size, dtype=np.float32)
                        for goal in self.goals:
                            layer[*goal] = 1.0
                    case ImageLayer.ACCESSIBLE:
                        layer = np.ones(self.grid_size, dtype=np.float32)
                        for ag in self.agents:
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

        if self.image_observation_directional:
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

    def _get_default_obs(self, agent: Agent):
        min_x = agent.pos.x - self.sensor_range
        max_x = agent.pos.x + self.sensor_range + 1

        min_y = agent.pos.y - self.sensor_range
        max_y = agent.pos.y + self.sensor_range + 1

        # sensors
        if (
            (min_x < 0)
            or (min_y < 0)
            or (max_x > self.grid_size[0])
            or (max_y > self.grid_size[1])
        ):
            padded_agents = np.pad(
                self.grid[_LAYER_AGENTS], self.sensor_range, mode="constant"
            )
            padded_shelfs = np.pad(
                self.grid[_LAYER_SHELVES], self.sensor_range, mode="constant"
            )
            # + self.sensor_range due to padding
            min_x += self.sensor_range
            max_x += self.sensor_range
            min_y += self.sensor_range
            max_y += self.sensor_range

        else:
            padded_agents = self.grid[_LAYER_AGENTS]
            padded_shelfs = self.grid[_LAYER_SHELVES]

        agents = padded_agents[min_x:max_x, min_y:max_y].reshape(-1)
        shelfs = padded_shelfs[min_x:max_x, min_y:max_y].reshape(-1)

        if self.fast_obs:
            # write flattened observations
            flatdim = gym.spaces.flatdim(self.observation_space[agent.id - 1])
            obs = _VectorWriter(flatdim)

            pos = (
                agent.pos.normalise(*self.grid_size)
                if self.normalised_coordinates
                else agent.pos
            )

            obs.write([*pos, int(agent.carrying_shelf is not None)])
            direction = np.zeros(4)
            direction[agent.dir.value] = 1.0
            obs.write(direction)
            obs.write([int(self.layout.is_highway(agent.pos))])

            # 'has_agent': MultiBinary(1),
            # 'direction': Discrete(4),
            # 'local_message': MultiBinary(2)
            # 'has_shelf': MultiBinary(1),
            # 'shelf_requested': MultiBinary(1),

            for i, (id_agent, id_shelf) in enumerate(zip(agents, shelfs)):
                if id_agent == 0:
                    # no agent, direction, or message
                    obs.write([0.0])  # no agent present
                    obs.write([1.0, 0.0, 0.0, 0.0])  # agent direction
                    obs.skip(self.msg_bits)  # agent message
                else:
                    obs.write([1.0])  # agent present
                    direction = np.zeros(4)
                    direction[self.agents[id_agent - 1].dir.value] = 1.0
                    obs.write(direction)  # agent direction as onehot
                    if self.msg_bits > 0:
                        obs.write(self.agents[id_agent - 1].message)  # agent message
                if id_shelf == 0:
                    obs.write([0.0, 0.0])  # no shelf or requested shelf
                else:
                    obs.write(
                        [1.0, int(self.shelves[id_shelf - 1] in self.request_queue)]
                    )  # shelf presence and request status
            return obs.vector

        # write dictionary observations
        obs = {}
        pos = (
            agent.pos.normalise(*self.grid_size)
            if self.normalised_coordinates
            else agent.pos
        )
        # --- self data
        obs["self"] = {
            "location": np.array(pos, dtype=np.int32),
            "carrying_shelf": [int(agent.carrying_shelf is not None)],
            "direction": agent.dir.value,
            "on_highway": [int(self.layout.is_highway(agent.pos))],
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
                obs["sensors"][i]["direction"] = self.agents[id_ - 1].dir.value
                obs["sensors"][i]["local_message"] = (
                    self.agents[id_ - 1].message if self.msg_bits > 0 else None
                )

        # find neighboring shelfs:
        for i, id_ in enumerate(shelfs):
            if id_ == 0:
                obs["sensors"][i]["has_shelf"] = [0]
                obs["sensors"][i]["shelf_requested"] = [0]
            else:
                obs["sensors"][i]["has_shelf"] = [1]
                obs["sensors"][i]["shelf_requested"] = [
                    int(self.shelves[id_ - 1] in self.request_queue)
                ]

        return obs

    def _make_obs(self, agent: Agent):
        if self.image_obs:
            return self._make_img_obs(agent)
        elif self.image_dict_obs:
            image_obs = self._make_img_obs(agent)
            feature_obs = _VectorWriter(
                self.observation_space[agent.id - 1]["features"].shape[0]
            )
            direction = np.zeros(4)
            direction[agent.dir.value] = 1.0
            feature_obs.write(direction)
            feature_obs.write(
                [
                    int(self.layout.is_highway(agent.pos)),
                    int(agent.carrying_shelf is not None),
                ]
            )
            return {
                "image": image_obs,
                "features": feature_obs.vector,
            }
        else:
            return self._get_default_obs(agent)

    def _get_info(self):
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

        return tuple([self._make_obs(agent) for agent in self.agents]), self._get_info()

    def step(
        self, actions: list[Action]
    ) -> tuple[list[np.ndarray], list[float], bool, bool, dict]:
        assert len(actions) == len(self.agents)

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
                agent.carrying_shelf
                and start != target
                and self.grid[_LAYER_SHELVES, *target]
                and not (
                    self.grid[_LAYER_AGENTS, *target]
                    and self.agents[
                        self.grid[_LAYER_AGENTS, *target] - 1
                    ].carrying_shelf
                )
            ):
                # there's a standing shelf at the target location
                # our agent is carrying a shelf so there's no way
                # this movement can succeed. Cancel it.
                agent.req_action = Action.NOOP
                G.add_edge(start, start)
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
            assert agent.req_action == Action.FORWARD
            agent.req_action = Action.NOOP

        rewards = np.zeros(self.n_agents)

        for agent in self.agents:
            agent.prev_pos = agent.pos

            if agent.req_action == Action.FORWARD:
                agent.pos = agent.req_location(self.grid_size)
                if agent.carrying_shelf:
                    agent.carrying_shelf.pos = agent.pos
            elif agent.req_action in [Action.LEFT, Action.RIGHT]:
                agent.dir = agent.req_direction()
            elif agent.req_action == Action.TOGGLE_LOAD and not agent.carrying_shelf:
                shelf_id = self.grid[_LAYER_SHELVES, *agent.pos]
                if shelf_id:
                    agent.carrying_shelf = self.shelves[shelf_id - 1]
            elif agent.req_action == Action.TOGGLE_LOAD and agent.carrying_shelf:
                if not self.layout.is_highway(agent.pos):
                    agent.carrying_shelf = None
                    if agent.has_delivered and self.reward_type == RewardType.TWO_STAGE:
                        rewards[agent.id - 1] += 0.5

                    agent.has_delivered = False

        self._recalc_grid()

        shelf_delivered = False
        for goal in self.goals:
            shelf_id = self.grid[_LAYER_SHELVES, *goal]
            if not shelf_id:
                continue
            shelf = self.shelves[shelf_id - 1]

            if shelf not in self.request_queue:
                continue
            # a shelf was successfully delived.
            shelf_delivered = True
            # remove from queue and replace it
            candidates = [s for s in self.shelves if s not in self.request_queue]
            new_request = self.np_random.choice(candidates)
            self.request_queue[self.request_queue.index(shelf)] = new_request
            # also reward the agents
            if self.reward_type == RewardType.GLOBAL:
                rewards += 1
            elif self.reward_type == RewardType.INDIVIDUAL:
                agent_id = self.grid[_LAYER_AGENTS, *goal]
                rewards[agent_id - 1] += 1
            elif self.reward_type == RewardType.TWO_STAGE:
                agent_id = self.grid[_LAYER_AGENTS, *goal]
                self.agents[agent_id - 1].has_delivered = True
                rewards[agent_id - 1] += 0.5

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

        new_obs = tuple([self._make_obs(agent) for agent in self.agents])
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
            layers = []
            for layer_type in image_layers:
                if layer_type == ImageLayer.SHELVES:
                    layer = self.grid[_LAYER_SHELVES].copy().astype(np.float32)
                    # set all occupied shelf cells to 1.0 (instead of shelf ID)
                    layer[layer > 0.0] = 1.0
                elif layer_type == ImageLayer.REQUESTS:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for requested_shelf in self.request_queue:
                        layer[*requested_shelf.pos] = 1.0
                elif layer_type == ImageLayer.AGENTS:
                    layer = self.grid[_LAYER_AGENTS].copy().astype(np.float32)
                    # set all occupied agent cells to 1.0 (instead of agent ID)
                    layer[layer > 0.0] = 1.0
                elif layer_type == ImageLayer.AGENT_DIRECTION:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        agent_direction = ag.dir.value + 1
                        layer[*ag.pos] = float(agent_direction)
                elif layer_type == ImageLayer.AGENT_LOAD:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        if ag.carrying_shelf is not None:
                            layer[*ag.pos] = 1.0
                elif layer_type == ImageLayer.GOALS:
                    layer = np.zeros(self.grid_size, dtype=np.float32)
                    for goal in self.goals:
                        layer[*goal] = 1.0
                elif layer_type == ImageLayer.ACCESSIBLE:
                    layer = np.ones(self.grid_size, dtype=np.float32)
                    for ag in self.agents:
                        layer[*ag.pos] = 0.0
                else:
                    raise ValueError(f"Unknown image layer type: {layer_type}")
                layers.append(layer)
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
