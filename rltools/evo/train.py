#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------
# Author:        sebastian.piec@
# Last modified: 2019
# ----------------------------------------------------------------------

"""
"""

import copy
import sys
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

# TODO
# ----------------------------------------------------------------------
def simple_loop(agent, environment, config, callbacks):
    """
    """
    state_size, n_actions = agent.state_size, agent.n_actions           # temp TODO 

    noise_range = (0.01, 0.0001)
    pool_size = 60
    n_parents = 15
    n_elite = 2
    n_generations = 20              # or other stopping criteria: time budget, avg fitness, plateu in fitness improvement?

    # TODO, make sure that the nets are randomly initialized
    pool = [CartAgent(state_size, n_actions) for _ in range(pool_size)] 

    for gen_idx in range(n_generations):
        sys.stderr.write("=" * 10 + "Generation {}".format(gen_idx) + "=" * 10 + "\n")
        
        pool.sort(key=lambda v: v.fitness, reverse=True)
        rewards = [p.fitness for p in pool[:n_parents]]

        # display average reward per pool of NN agents
        print("AVG parents reward:", np.mean(rewards))

        # TODO: keep elite always, 


        # new population
        prev_pool = pool
        pool = [pool[0]]

        # evaluate all agents in the pool in parallel

        noise_std = noise_range[0] #- gen_idx * (noise_range[0] - noise_range[1]) / (n_generations - 1)

        for agent_idx in range(pool_size):
            idx = np.random.randint(0, n_parents)

            agent = prev_pool[idx]
            new_agent = agent.mutate(noise_std)
            new_agent.evaluate(environment, max_reward=1.0)            # recalculate agent's fitness

            pool.append(new_agent)
            
            #for callback in callbacks:
            #    callback((mean_reward, agent.action_selector.epsilon, episode_idx))
            sys.stderr.write("Agent {}, fitness {:.2f}\n".format(agent_idx, new_agent.fitness))

        print("Generation {} finished!".format(gen_idx))

        pool_fitness = np.mean([v.fitness for v in pool])
        sys.stderr.write("AVG fitness of the pool {:.2f}\n".format(pool_fitness))

    # best agent in the last generation...
    pool.sort(key=lambda v: v.fitness, reverse=True)
    best_agent, best_fitness = pool[0], pool[0].fitness

    print("Best fitness", best_fitness, "agent type:", type(best_agent))

    return best_agent


    # return K best, use ensemble decision in production?




# ----------------------------------------------------------------------
#def optimize_hyperparams():




    """
# ----------------------------------------------------------------------
def evaluate(environment, net):
    max_episode_reward = 1  #20
    max_steps = int(10e+5)          # max steps allowed per episode
    episode_reward = 0.0

    # don't render? TODO
    episode_reward = 0.0
    state = environment.reset()

    # NN fitness based on the FULL episode? TODO
    for step_idx in range(max_steps):
            reward, action_idx, new_state, is_done = environment.step(state, net)
        
            #agent.memorize(state, action_idx, reward, new_state, is_done)

            #if step_idx == 0:
            #net.training_step()               # ??? TODO

            state = new_state
            episode_reward += reward

            if is_done or episode_reward >= max_episode_reward:
                print("Last reward:", reward, ", episode reward:", episode_reward, ", is_done:", is_done)
                break

    return episode_reward


# ----------------------------------------------------------------------
def mutate_parent(net, noise_std=0.01):
    new_net = copy.deepcopy(net)

    # 
    for p in new_net.parameters():
        noise_t = torch.tensor(np.random.normal(size=p.data.size()).astype(np.float32))
        p.data += noise_std * noise_t

    # mirroring

    return new_net
    """


