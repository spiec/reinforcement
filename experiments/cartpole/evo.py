#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:   sebastian.piec@
# Modified: 2019, March 13
# ----------------------------------------------------------------------

import copy
import sys
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from rltools.evo.agent import EvoAgent

# ----------------------------------------------------------------------
class CartNet(nn.Module):

    def __init__(self, state_size, n_actions):
        super(CartNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=1)                       # N-dim action vector
        )
        print(self)

    def forward(self, x):
        return self.fc(x)

# ----------------------------------------------------------------------
class CartAgent(EvoAgent):

    def __init__(self, state_size, n_actions, architecture=CartNet):
        super(CartAgent, self).__init__(state_size, n_actions, architecture)
        self.reinitialize()

# ----------------------------------------------------------------------
def training_loop(agent, environment, config, callbacks):
    state_size, n_actions = agent.state_size, agent.n_actions

    n_parents = int(config.pool_size * 0.2)
    elite_size = max(2, int(config.pool_size * 0.04))
    noise_range = (0.01, 0.0001)
    
    agent_pool = [CartAgent(state_size, n_actions) for _ in range(config.pool_size)] 

    for gen_idx in range(config.n_generations):         # try other stopping criteria: time budget, avg fitness, plateu in fitness improvement?
        sys.stderr.write("=" * 10 + "Generation {}".format(gen_idx) + "=" * 10 + "\n")
        
        agent_pool.sort(key=lambda v: v.fitness, reverse=True)
        rewards = [p.fitness for p in agent_pool[:n_parents]]
        print("AVG parents reward:", np.mean(rewards))

        noise_std = noise_level(noise_range, gen_idx, config.n_generations)

        parent_indices = np.zeros(config.pool_size)
        parent_indices[:elite_size] = np.arange(elite_size)         # elite should always survive?

        # the rest select with probability proprtional to agent's fitness
        selection_p = np.arange(n_parents)
        selection_p = selection_p / np.sum(selection_p)
        parent_indices = np.random.choice(range(n_parents), config.pool_size - elite_size, p=selection_p)

        agent_pool = next_generation(agent_pool, environment, parent_indices=parent_indices, noise=noise_std)

        pool_fitness = np.mean([v.fitness for v in agent_pool])
        sys.stderr.write("AVG fitness of the pool {:.2f}\n".format(pool_fitness))

        agent_pool.sort(key=lambda v: v.fitness, reverse=True)
        best_agent, best_fitness = agent_pool[0], agent_pool[0].fitness

        best_agent.save(config.agent)

    # best agent in the last generation...
    agent_pool.sort(key=lambda v: v.fitness, reverse=True)
    best_agent, best_fitness = agent_pool[0], agent_pool[0].fitness
    print("Best fitness", best_fitness, "agent type:", type(best_agent))

    return best_agent

# ----------------------------------------------------------------------
def next_generation(agent_pool, environment, parent_indices=[], noise=0.01):
    new_pool = []
    for idx in parent_indices:
        agent = agent_pool[idx]
        new_agent = agent.mutate(noise)         # generates new, possibly better, agent
        new_agent.evaluate(environment, max_reward=1000000)
        new_pool.append(new_agent)
        sys.stderr.write("Agent {}, fitness {:.2f}\n".format(idx, new_agent.fitness))
    return new_pool

# ----------------------------------------------------------------------
def noise_level(noise_range, generation, max_generations):
    # decaying noise?
    return noise_range[0] #- gen_idx * (noise_range[0] - noise_range[1]) / (n_generations - 1)