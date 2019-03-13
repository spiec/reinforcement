#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 13
# ----------------------------------------------------------------------

"""
Usage:
    python play.py --agent_file output/pongdqn_v1.pt
"""

import torch

import config
from common import display_versions, initial_seed, timer
from game import PongGame

from dqn import PongAgent
#from evo import PongAgent

# ----------------------------------------------------------------------
def main():
    config.read_args()

    initial_seed(config.seed)
    display_versions()
    
    game = PongGame(render=True)
    state_size, n_actions = game.state_size(), game.n_actions()

    agent = PongAgent(state_size, n_actions)
    agent.load(config.agent)
    #print(agent.net)

    n_episodes = 10

    for episode_idx in range(n_episodes):
        episode_return = 0.0
        state = game.reset()

        while True:
            action_idx, expected_reward = agent.action(state, allow_random=False)        # left/right/do nothing
            reward, new_state, is_done = game.player_action(action_idx)   
            episode_return += reward
            state = new_state

            if is_done or episode_return >= 1:
                break

        print("Episode {} finished, return: {}".format(episode_idx, episode_return))

# ----------------------------------------------------------------------
if __name__ == "__main__":
    with timer("Pong RL"):
        main()