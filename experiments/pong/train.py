#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 13
# ----------------------------------------------------------------------

"""
Usage:
    python train.py --episodes 1000
"""

from rltools.utils.callbacks import ModelSaver, TrainingLogger

import config
from common import display_versions, initial_seed, timer
from environment import PongEnvironment

# ----------------------------------------------------------------------
def train_agent(mode="dqn"):
    env = PongEnvironment(config.seed)
    state_size, n_actions = env.info()
    
    if mode == "dqn":
        from dqn import PongAgent, training_loop
        agent = PongAgent(state_size, n_actions)
    elif mode == "evo":
        from evo import PongAgent, training_loop
        agent = PongAgent(state_size, n_actions)

    callbacks = [
        TrainingLogger(config.training_log),
        ModelSaver(agent, config.agent),
    ]

    agent = training_loop(agent, env, config, callbacks)
    return agent

# ----------------------------------------------------------------------
def main():
    config.read_args()

    initial_seed(config.seed)
    display_versions()

    agent = train_agent("dqn")          # dqn, evo, pg
    agent.save(config.agent)

    print("All OK")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    with timer("Pong RL"):
        main()