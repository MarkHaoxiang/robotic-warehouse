from typing import Dict, Tuple, List, Optional

import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv

from .warehouse import Warehouse

# ID are str(integers), which represent the agent.id (agent idx+1) in env.agents.
# Set to str for compatability with TorchRL.
AgentID = str
# TODO: Refactor Action object to include the message bits.
ActionType = object
ObsType = np.ndarray


def to_agentid_dict(data: List):
    return {str(i + 1): x for i, x in enumerate(data)}


class PettingZooWrapper(ParallelEnv):
    """Wraps a Warehouse Env object to be compatible with the PettingZoo ParallelEnv API."""

    def __init__(self, env: Warehouse):
        super().__init__()
        self._env = env
        self.agents: List[AgentID] = []
        self.possible_agents: List[AgentID] = []
        self.observation_spaces = self.action_spaces = {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = self._env.reset(seed, options)
        obs = to_agentid_dict(obs)
        # Reset agents and spaces
        self.agents = self.possible_agents = [
            str(agent.id) for agent in self._env.agents
        ]
        self.observation_spaces = {
            agent_id: self.observation_space(agent_id) for agent_id in self.agents
        }
        self.action_spaces = {
            agent_id: self.action_space(agent_id) for agent_id in self.agents
        }

        for agent_id in self.agents:
            info[agent_id] = info[int(agent_id)]

        return obs, info

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> Tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        # Unwrap to list of actions
        actions_unwrapped = [(int(id_) - 1, action) for id_, action in actions.items()]
        actions_unwrapped.sort(key=lambda x: x[0])
        actions_unwrapped = [x[1] for x in actions_unwrapped]
        assert len(actions_unwrapped) == self._env.n_agents, (
            f"Incorrect number of actions provided. Expected {self._env.n_agents} but got {len(actions_unwrapped)}"
        )

        # Step inner environment
        obs, rewards, terminated, truncated, info = self._env.step(actions_unwrapped)

        # Transform to PettingZoo output
        obs = to_agentid_dict(obs)
        rewards = to_agentid_dict(rewards)
        if terminated or truncated:
            self.agents = []  # PettingZoo requires agents to be removed
        terminated = to_agentid_dict([terminated for _ in range(self._env.n_agents)])
        truncated = to_agentid_dict([truncated for _ in range(self._env.n_agents)])

        for agent_id in self.agents:
            info[agent_id] = info[int(agent_id)]

        return obs, rewards, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    def state(self):
        return self._env.get_global_image()

    def observation_space(self, agent: AgentID) -> gym.spaces.Space:
        space = self._env.observation_space
        assert isinstance(space, gym.spaces.Tuple)
        return space[int(agent) - 1]

    def action_space(self, agent: AgentID) -> gym.spaces.Space:
        space = self._env.action_space
        assert isinstance(space, gym.spaces.Tuple)
        return space[int(agent) - 1]

    def __getattr__(self, name: str):
        return getattr(self._env, name)
