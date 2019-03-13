#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 13
# ----------------------------------------------------------------------

"""Play game using trained agent.
Usage:
    python play.py --agent models/cartagent_v0.pt
"""

import sys
import numpy as np
import gym
import torch

from rltools.utils import GymWrapper

from common import initial_seed, timer
import config
from evo import CartAgent

# ----------------------------------------------------------------------
def main():
    config.read_args()
    initial_seed(config.seed)

    gymenv = gym.make("CartPole-v1")
    gymenv._max_episode_steps = 2000
    env = GymWrapper(gymenv, seed=config.seed, render=True)
    
    state_size, n_actions = env.info()

    agent = CartAgent(state_size, n_actions)
    agent.load(config.agent)

    n_episodes = 10
    returns = []

    for episode_idx in range(n_episodes):
        episode_return = 0.0
        state = env.reset()

        while True:
            reward, action, new_state, is_done = env.step(state, agent)
            state = new_state
            episode_return += reward

            if is_done:
                returns.append(episode_return)
                break

        sys.stderr.write("Episode {} finished, return: {}\n".format(episode_idx, episode_return))

    print("Average reward per episode: {:.2f}".format(np.mean(returns)))

# ----------------------------------------------------------------------
if __name__ == "__main__":
    with timer("CartPole play"):
        main()