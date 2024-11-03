#!/usr/bin/env python3
"""a script to train an agent to play Atari's Breakout
using Deep Q-Learning and save the trained policy."""
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

env = gym.make('BreakoutNoFrameskip-v4')
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
memory = SequentialMemory(limit=10000, window_length=4)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory,
               target_model_update=1e-2, policy=policy)
dqn.compile(keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=10000, visualize=False, verbose=1, batch_size=2)
dqn.save_weights('policy.h5', overwrite=True)
