#!/usr/bin/env python3
"""a function that performs SARSA(λ)"""
import numpy as np


def sarsa_lambtha(
        env,
        Q,
        lambtha,
        episodes=5000,
        max_steps=100,
        alpha=0.1,
        gamma=0.99,
        epsilon=1,
        min_epsilon=0.1,
        epsilon_decay=0.05):
    """Returns: Q, the updated Q table"""
    def eps_greedy(state, Q, epsilon):
        """selects action using epsilon greedy policy"""
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(Q[state])
    for ep in range(episodes):
        state, _ = env.reset()
        action = eps_greedy(state, Q, epsilon)
        eligibility_trace = np.zeros_like(Q)
        for step in range(max_steps):
            next_state, reward, done, _, _ = env.step(action)
            next_action = eps_greedy(next_state, Q, epsilon)
            td_error = reward + gamma * \
                Q[next_state, next_action] - Q[state, action]
            eligibility_trace[state, action] += 1
            Q += alpha * td_error * eligibility_trace
            eligibility_trace *= gamma * lambtha
            state, action = next_state, next_action
            if done:
                break
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay * ep))
    return Q
