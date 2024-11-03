#!/usr/bin/env python3
"""a script that loads the saved policy and lets the agent play Breakout."""
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

env = gym.make('BreakoutNoFrameskip-v4', render_mode='human')
env = AtariPreprocessing(env)
env = FrameStack(env, 4)
model = Sequential()
model.add(
    Convolution2D(
        32, (3, 3), strides=(
            1, 1), activation='relu', padding='same', input_shape=(
                84, 84, 4)))
model.add(Convolution2D(64, (3, 3), activation='relu', padding='valid'))
model.add(Convolution2D(64, (3, 3), activation='relu', padding='valid'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
model.load_weights('policy.h5')
greedy_policy = GreedyQPolicy()
dqn_test = DQNAgent(
    model=model,
    nb_actions=env.action_space.n,
    policy=greedy_policy,
    test_policy=greedy_policy,
    processor=None)
dqn_test.test(env, nb_episodes=5, visualize=True)
