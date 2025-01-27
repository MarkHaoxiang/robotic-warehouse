import os
import sys

import gymnasium as gym
import numpy as np
import pytest
from expecttest import assert_expected_inline

from rware.layout import Layout
from rware.observation import ObservationType
from rware.utils.typing import Direction, ImageLayer
from rware.warehouse import (
    Warehouse,
    Action,
    RewardType,
    ImageLayer,
)


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)


@pytest.fixture
def env_single_agent():
    env = Warehouse(3, 8, 3, 1, 0, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    return env


@pytest.fixture
def env_0():
    env = Warehouse(3, 8, 3, 1, 0, 1, 5, 10, None, RewardType.GLOBAL)
    env.reset()

    env.agents[0].x = 4  # should place it in the middle (empty space)
    env.agents[0].y = 27
    env.agents[0].dir = Direction.DOWN

    env.shelves[0].x = 4
    env.shelves[0].y = 27

    env.agents[0].carried_shelf = env.shelves[0]

    env.request_queue[0] = env.shelves[0]
    env._recalc_grid()
    return env


def test_env_layout_from_params():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
    )
    env.reset()
    layout = str(env.get_global_image())
    assert_expected_inline(
        layout,
        """\
[[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 1. 1. 1. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0.]
  [0. 1. 1. 1. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]]""",
    )


def test_env_layout_from_str():
    layout = """
.......
...x...
..x.x..
.x...x.
..x.x..
...x...
.g...g.
"""
    env = Warehouse(layout=layout)
    env.reset()
    layout = str(env.get_global_image())
    assert_expected_inline(
        layout,
        """\
[[[0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 1. 0. 0. 0.]
  [0. 0. 1. 0. 1. 0. 0.]
  [0. 1. 0. 0. 0. 1. 0.]
  [0. 0. 1. 0. 1. 0. 0.]
  [0. 0. 0. 1. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 1.]
  [0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 1.]
  [0. 0. 0. 0. 0. 0. 0.]]]""",
    )


def test_env_layout_from_image(env_0: Warehouse):
    env_0.reset()
    layers = [
        ImageLayer.SHELVES,
        ImageLayer.GOALS,
        ImageLayer.AGENTS,
        ImageLayer.AGENT_DIRECTION,
    ]
    baseline_grid = env_0.get_global_image(image_layers=layers)
    layout = Layout.from_image(baseline_grid, layers)
    env = Warehouse(layout=layout)
    env.reset()
    grid = env.get_global_image(image_layers=layers)

    assert str(grid) == str(baseline_grid)

    layout = str(env.get_global_image(recompute=True))
    assert_expected_inline(
        layout,
        """\
[[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1.
   1. 1. 1. 1. 0. 0.]
  [0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1.
   1. 1. 1. 1. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1.
   1. 1. 1. 1. 0. 0.]
  [0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1. 1. 1. 1. 1. 0. 1. 1. 1. 1.
   1. 1. 1. 1. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]]

 [[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 1.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 1.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
   0. 0. 0. 0. 0. 0.]]]""",
    )


def test_grid_size():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=1,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    assert env.grid_size == (4, 14)
    env = Warehouse(
        shelf_columns=3,
        column_height=3,
        shelf_rows=3,
        n_agents=1,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    assert env.grid_size == (10, 14)


def test_action_space_0():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=2,
        msg_bits=0,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    env.reset()
    assert env.action_space == gym.spaces.Tuple(2 * (gym.spaces.Discrete(len(Action)),))
    env.step(env.action_space.sample())


def test_action_space_1():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=2,
        msg_bits=1,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    env.reset()
    assert env.action_space == gym.spaces.Tuple(
        2 * (gym.spaces.MultiDiscrete([len(Action), 2]),)
    )
    env.step(env.action_space.sample())


def test_action_space_2():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=2,
        msg_bits=2,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    env.reset()
    assert env.action_space == gym.spaces.Tuple(
        2 * (gym.spaces.MultiDiscrete([len(Action), 2, 2]),)
    )
    env.step(env.action_space.sample())


def test_action_space_3():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=10,
        msg_bits=5,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
    )
    env.reset()
    assert env.action_space == gym.spaces.Tuple(
        10 * (gym.spaces.MultiDiscrete([len(Action), *5 * (2,)]),)
    )
    env.step(env.action_space.sample())


@pytest.mark.parametrize(
    "observation_type",
    [
        ObservationType.DICT,
        ObservationType.FLATTENED,
        ObservationType.IMAGE,
        ObservationType.IMAGE_DICT,
        ObservationType.IMAGE_LAYOUT,
    ],
)
def test_obs_space_contains(observation_type: ObservationType):
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=10,
        msg_bits=5,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        observation_type=observation_type,
        reward_type=RewardType.GLOBAL,
    )
    obs, _ = env.reset()
    for _ in range(100):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)


def test_obs_space_0():
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=10,
        msg_bits=5,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=None,
        reward_type=RewardType.GLOBAL,
        observation_type=ObservationType.DICT,
    )
    obs, _ = env.reset()
    for i in range(env.unwrapped.n_agents):
        for key in env.observation_space[i]["self"].keys():
            if key == "direction":
                # direction is not considered 'contained' by gym if onehot
                continue
            else:
                print(
                    i,
                    "self",
                    key,
                    env.observation_space[i]["self"][key],
                    obs[i]["self"][key],
                )
                assert env.observation_space[i]["self"][key].contains(
                    obs[i]["self"][key]
                ), f"{obs[i]['self'][key]} is not contained in {env.observation_space[i]['self'][key]}"
        for j in range(len(env.observation_space[i]["sensors"])):
            for key in env.observation_space[i]["sensors"][j].keys():
                if key == "direction":
                    # direction is not considered 'contained' by gym if onehot
                    continue
                else:
                    print(
                        i,
                        "sensors",
                        key,
                        env.observation_space[i]["sensors"][j][key],
                        obs[i]["sensors"][j][key],
                    )
                    assert env.observation_space[i]["sensors"][j][key].contains(
                        obs[i]["sensors"][j][key]
                    ), f"{obs[i]['sensors'][j][key]} is not contained in {env.observation_space[i]['sensors'][j][key]}"
    obs, _, _, _, _ = env.step(env.action_space.sample())
    for i in range(env.unwrapped.n_agents):
        for key in env.observation_space[i]["self"].keys():
            if key == "direction":
                # direction is not considered 'contained' by gym if onehot
                continue
            else:
                print(
                    i,
                    "self",
                    key,
                    env.observation_space[i]["self"][key],
                    obs[i]["self"][key],
                )
                assert env.observation_space[i]["self"][key].contains(
                    obs[i]["self"][key]
                ), f"{obs[i]['self'][key]} is not contained in {env.observation_space[i]['self'][key]}"
        for j in range(len(env.observation_space[i]["sensors"])):
            for key in env.observation_space[i]["sensors"][j].keys():
                if key == "direction":
                    # direction is not considered 'contained' by gym if onehot
                    continue
                else:
                    print(
                        i,
                        "sensors",
                        key,
                        env.observation_space[i]["sensors"][j][key],
                        obs[i]["sensors"][j][key],
                    )
                    assert env.observation_space[i]["sensors"][j][key].contains(
                        obs[i]["sensors"][j][key]
                    ), f"{obs[i]['sensors'][j][key]} is not contained in {env.observation_space[i]['sensors'][j][key]}"


# def test_obs_space_2():
#     env = Warehouse(
#         shelf_columns=1,
#         column_height=3,
#         shelf_rows=3,
#         n_agents=10,
#         msg_bits=5,
#         sensor_range=1,
#         request_queue_size=5,
#         max_inactivity_steps=None,
#         max_steps=None,
#         reward_type=RewardType.GLOBAL,
#     )
#     obs, _ = env.reset()
#     for s, o in zip(env.observation_space, obs):
#         assert len(gym.spaces.flatten(s, o)) == env._obs_length


def test_inactivity_0(env_0):
    env = env_0
    for i in range(9):
        _, _, done, _, _ = env.step([Action.NOOP])
        assert not done
    _, _, done, _, _ = env.step([Action.NOOP])
    assert done


def test_inactivity_1(env_0):
    env = env_0
    for i in range(4):
        _, _, done, _, _ = env.step([Action.NOOP])
        assert not done
    _, reward, _, _, _ = env.step([Action.FORWARD])
    assert reward[0] == pytest.approx(1.0)
    for i in range(9):
        _, _, done, _, _ = env.step([Action.NOOP])
        assert not done

    _, _, done, _, _ = env.step([Action.NOOP])
    assert done


@pytest.mark.parametrize("time_limit,", [1, 100, 200])
def test_time_limit(time_limit):
    env = Warehouse(
        shelf_columns=1,
        column_height=3,
        shelf_rows=3,
        n_agents=10,
        msg_bits=5,
        sensor_range=1,
        request_queue_size=5,
        max_inactivity_steps=None,
        max_steps=time_limit,
        reward_type=RewardType.GLOBAL,
    )
    env.reset()

    for _ in range(time_limit - 1):
        _, _, done, _, _ = env.step(env.action_space.sample())
        assert not done

    _, _, done, _, _ = env.step(env.action_space.sample())
    assert done


def test_inactivity_2(env_0):
    env = env_0
    for i in range(9):
        _, _, done, _, _ = env.step([Action.NOOP])
        assert not done
    _, _, done, _, _ = env.step([Action.NOOP])
    assert done
    env.reset()
    for i in range(9):
        _, _, done, _, _ = env.step([Action.NOOP])
        assert not done
    _, _, done, _, _ = env.step([Action.NOOP])
    assert done


# @pytest.mark.parametrize("n_agents", [2, 3])
# @pytest.mark.parametrize("msg_bits", [0, 2])
# def test_fast_obs(n_agents: int, msg_bits: int):
#     env = Warehouse(
#         shelf_columns=3,
#         column_height=8,
#         shelf_rows=3,
#         n_agents=n_agents,
#         msg_bits=msg_bits,
#         observation_type=ObservationType.DICT,
#     )
#     env.reset(seed=0)

#     slow_obs_space = env.observation_space

#     for _ in range(10):
#         env._use_slow_obs()
#         slow_obs = [env._make_obs(agent) for agent in env.agents]
#         env._use_fast_obs()
#         fast_obs = [env._make_obs(agent) for agent in env.agents]

#         assert len(fast_obs) == len(slow_obs) == env.n_agents

#         flattened_slow = [
#             gym.spaces.flatten(osp, obs) for osp, obs in zip(slow_obs_space, slow_obs)
#         ]
#         for slow, fast in zip(flattened_slow, fast_obs):
#             slow, fast = slow.tolist(), fast.tolist()
#             slow.sort()
#             fast.sort()
#             assert fast == slow

#         env.step(env.action_space.sample())


def test_reproducibility(env_0):
    env = env_0
    episodes_per_seed = 5
    for seed in range(5):
        obss1 = []
        grid1 = []
        highways1 = []
        request_queue1 = []
        player_pos1 = []
        player_carrying1 = []
        player_has_delivered1 = []
        env.seed(seed)
        for _ in range(episodes_per_seed):
            obss, _ = env.reset()
            obss1.append(np.array(obss).copy())
            grid1.append(env.unwrapped.grid.copy())
            highways1.append(env.unwrapped.layout.highways.copy())
            request_queue1.append(
                np.array([shelf.id for shelf in env.unwrapped.request_queue])
            )
            player_pos1.append([p.pos for p in env.unwrapped.agents])
            player_carrying1.append([p.carried_shelf for p in env.unwrapped.agents])
            player_has_delivered1.append(
                [p.has_delivered for p in env.unwrapped.agents]
            )

        obss2 = []
        grid2 = []
        highways2 = []
        request_queue2 = []
        player_pos2 = []
        player_carrying2 = []
        player_has_delivered2 = []
        env.seed(seed)
        for _ in range(episodes_per_seed):
            obss, _ = env.reset()
            obss2.append(np.array(obss).copy())
            grid2.append(env.unwrapped.grid.copy())
            highways2.append(env.unwrapped.layout.highways.copy())
            request_queue2.append(
                np.array([shelf.id for shelf in env.unwrapped.request_queue])
            )
            player_pos1.append([p.pos for p in env.unwrapped.agents])
            player_carrying2.append([p.carried_shelf for p in env.unwrapped.agents])
            player_has_delivered2.append(
                [p.has_delivered for p in env.unwrapped.agents]
            )

        for i, (obs1, obs2) in enumerate(zip(obss1, obss2)):
            assert np.array_equal(
                obs1, obs2
            ), f"Observations of env not identical for episode {i} with seed {seed}"
        for i, (g1, g2) in enumerate(zip(grid1, grid2)):
            assert np.array_equal(
                g1, g2
            ), f"Grid of env not identical for episode {i} with seed {seed}"
        for i, (h1, h2) in enumerate(zip(highways1, highways2)):
            assert np.array_equal(
                h1, h2
            ), f"Highways of env not identical for episode {i} with seed {seed}"
        for i, (rq1, rq2) in enumerate(zip(request_queue1, request_queue2)):
            assert np.array_equal(
                rq1, rq2
            ), f"Request queue of env not identical for episode {i} with seed {seed}"
        for i, (p1, p2) in enumerate(zip(player_pos1, player_pos2)):
            assert np.array_equal(
                p1, p2
            ), f"Player pos of env not identical for episode {i} with seed {seed}"
        for i, (pc1, pc2) in enumerate(zip(player_carrying1, player_carrying2)):
            assert np.array_equal(
                pc1, pc2
            ), f"Player carrying of env not identical for episode {i} with seed {seed}"
        for i, (pd1, pd2) in enumerate(
            zip(player_has_delivered1, player_has_delivered2)
        ):
            assert np.array_equal(
                pd1, pd2
            ), f"Player has delivered of env not identical for episode {i} with seed {seed}"
