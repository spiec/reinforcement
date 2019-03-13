#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 13
# ----------------------------------------------------------------------

seed = 999
agent = "output/pongdqn_v2.pt"

# dqn or pg
n_episodes = 100

# evolutionary based agent
pool_size = 80
n_generations = 10

training_log = "logs/tmp.log"

# ----------------------------------------------------------------------
def read_args():
    global seed, agent, n_episodes

    import argparse

    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-s", "--seed", type=int, default=seed,
                        help="random seed")
    parser.add_argument("-e", "--episodes", type=int, default=n_episodes,
                        help="# of episodes to train agent on")
    parser.add_argument("-a", "--agent", type=str, default=agent,
                        help="model/agent file")
    args = parser.parse_args()

    n_episodes = args.episodes
    seed = args.seed
    agent = args.agent