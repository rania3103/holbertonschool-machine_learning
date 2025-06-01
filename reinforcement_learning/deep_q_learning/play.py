#!/usr/bin/env python3
"""a script that loads the saved policy and lets the agent play Breakout."""
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory
import tensorflow as tf


def build_model(input_shape, num_actions):
    model = Sequential()
    model.add(
        Conv2D(
            32,
            (8,
             8),
            strides=4,
            activation='relu',
            input_shape=input_shape))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    return model


if __name__ == "__main__":
    print(tf.config.list_physical_devices('GPU'))
    env = gym.make('ALE/Breakout-v5', render_mode="human")
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True)
    env = FrameStack(env, 4)
    nb_actions = env.action_space.n
    input_shape = (84, 84, 4)
    model = build_model(input_shape, nb_actions)
    print(model.summary())
    memory = SequentialMemory(limit=1000000, window_length=4)
    # greedy policy for testing
    policy = GreedyQPolicy()
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=50000,
        target_model_update=10000,
        policy=policy)
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    dqn.load_weights('policy.h5')
    dqn.test(env, nb_episodes=5, visualize=True)
    env.close()
