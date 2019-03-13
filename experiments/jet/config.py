#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 11
# ----------------------------------------------------------------------

seed = 999
agent = "output/jetdqn_v1.pt"
training_log = "logs/training.log"

# dqn
n_episodes = 100

# evolutionary agents
pool_size = 120
n_generations = 40

# ----------------------------------------------------------------------
def read_args():
    global seed, agent, n_episodes

    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-s", "--seed", type=int, default=seed,
                        help="random seed")
    parser.add_argument("-e", "--episodes", type=int, default=n_episodes,
                        help="# of episodes")
    parser.add_argument("-a", "--agent", type=str, default=agent,
                        help="agent file")
    args = parser.parse_args()

    n_episodes = args.episodes
    seed = args.seed
    agent = args.agent
