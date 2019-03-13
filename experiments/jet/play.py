#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 11
# ----------------------------------------------------------------------

"""Play game using trained agent.

Usage:
    python play.py --agent output/jetdqn_v0.pt
"""

import sys
import numpy as np

from rltools.utils import GymWrapper

import config
from common import initial_seed, timer
from environment import JetEnvironment

from dqn import JetAgent

# ----------------------------------------------------------------------
def main():
    config.read_args()
    initial_seed(config.seed)

    env = JetEnvironment(render=True)
    state_size, n_actions = env.info()

    agent = JetAgent(state_size, n_actions)
    agent.load(config.agent)

    n_episodes = 100
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
    with timer("Jet play"):
        main()