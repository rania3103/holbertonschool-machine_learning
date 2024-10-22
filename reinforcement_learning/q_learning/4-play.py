#!/usr/bin/env python3
"""a function that has the trained agent play an episode"""
import numpy as np


def play(env, Q, max_steps=100):
    """Returns: The total rewards for the episode and
    a list of rendered outputs representing
    the board state at each step."""
    state = env.reset()[0]
    total_rewards = 0
    rendered_outputs = []
    rendered_outputs.append(env.render())
    for step in range(max_steps):
        action = np.argmax(Q[state])
        next_state, reward, done, truncated, info = env.step(action)
        total_rewards += reward
        rendered_outputs.append(env.render())
        if done:
            break
        state = next_state
    return total_rewards, rendered_outputs
