#!/usr/bin/env python3
"""a function that implements a full training."""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """returns all values of the score
    (sum of all rewards during one episode loop)"""
    weight = np.random.rand(env.observation_space.shape[0], env.action_space.n)
    scores = []
    for ep in range(nb_episodes):
        state, _ = env.reset()
        done = False
        ep_gradients = []
        ep_rewards = []
        while not done:
            action, grad = policy_gradient(state, weight)
            next_state, reward, done, _, _ = env.step(action)
            ep_rewards.append(reward)
            ep_gradients.append(grad)
            state = next_state
        result = []
        cum_reward = 0
        for r in reversed(ep_rewards):
            cum_reward = r + gamma * cum_reward
            result.insert(0, cum_reward)
        for t in range(len(ep_rewards)):
            weight += alpha * ep_gradients[t] * result[t]
        score = sum(ep_rewards)
        scores.append(score)
        print(f"Episode: {ep} Score: {score}")
    return scores
