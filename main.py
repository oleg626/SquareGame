import random

import numpy as np
import pygame
import sys
import gym
from environment import SquaresEnv
from SquareGameRender import SquareGameRenderer
from pretrain import ExpertDataSet, pretrain_agent
from torch.utils.data.dataset import random_split
from stable_baselines3 import PPO
import os
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
# It will check your custom environment and output additional warnings if needed

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


def check_runtime(myEnv):
    step_times = []
    done = False
    for i in range(1000):
        myEnv.reset()
        while not done:
            action = myEnv.action_space.sample()
            t1 = time.perf_counter()
            state, reward, done, info = myEnv.step(action)
            step_times.append(time.perf_counter() - t1)
    print(f'step takes average ns: {sum(step_times) / len(step_times) * 1000000}')

    step_times = []
    done = False
    for i in range(1000):
        myEnv.reset()
        while not done:
            action = myEnv.action_space.sample()
            t1 = time.perf_counter()
            state, reward, done, info = myEnv.step(action)
            step_times.append(time.perf_counter() - t1)
    print(f'step takes average ns: {sum(step_times) / len(step_times) * 1000000}')

    step_times = []
    done = False
    for i in range(1000):
        myEnv.reset()
        while not done:
            action = myEnv.action_space.sample()
            t1 = time.perf_counter()
            state, reward, done, info = myEnv.step(action)
            step_times.append(time.perf_counter() - t1)
    print(f'step takes average ns: {sum(step_times) / len(step_times) * 1000000}')


myEnv = SquaresEnv()
state = myEnv.reset()
game = SquareGameRenderer(myEnv, myEnv.get_obs(), myEnv.get_shape(), num_episodes=20)
game.start()
check_env(myEnv)
#check_runtime(myEnv)

# PRETRAIN SECTION
expert_observations = []
expert_actions = []
for i in range(0, 3):
    state = myEnv.reset()
    done = False
    temp = 0
    while not done:
        myEnv.render('human')
        # y = int(input("Enter Y:"))
        # x = int(input("Enter X:"))
        # action = [y, x]
        action = myEnv.action_space.sample()
        expert_observations.append(state)
        expert_actions.append(action)
        state, reward, done, info = myEnv.step(action)
# for i in range(0, len(expert_actions)):
#     print(expert_observations[i])
#     print(expert_actions[i])

np.savez_compressed(
    f"expert_data_{np.random.randint(1,10000)}",
    expert_actions=expert_actions,
    expert_observations=expert_observations,
)
expert_dataset = ExpertDataSet(expert_observations, expert_actions)

train_size = int(0.8 * len(expert_dataset))

test_size = len(expert_dataset) - train_size

train_expert_dataset, test_expert_dataset = random_split(
    expert_dataset, [train_size, test_size]
)
# end of PRETRAIN SECTION



# model = PPO.load('models/linear_obs_action_MD_lr_0.003_batch_128_clip_0.4_ent_0.01_484/446500000')
# average_rewards = []
# temp = 0
# done = False
# for num in range(4, 11):
#     myEnv.set_num_of_shapes(num)
#     for i in range(5):
#         state = myEnv.reset()
#         done = False
#         temp = 0
#         while not done:
#             action = model.predict(state)
#             state, reward, done, info = myEnv.step(action[0])
#             temp += reward
#         average_rewards.append(temp)
#     print(f'average reward: {sum(average_rewards) / len(average_rewards)}')

# for lr in [0.003]:
#     for batch in [128]:
#         for clip in [0.4]:
#             for ent in [0.01]:
#                 envType = 'linear_obs_action_MD_'
#                 run = envType + f'lr_{lr}_batch_{batch}_clip_{clip}_ent_{ent}_{np.random.randint(0, 1000)}'
#                 models_dir = f"models/{run}/"
#                 logdir = f"logs/{run}/"
#                 # Parallel environments
#                 #env = make_vec_env(SquaresEnv, n_envs=6)
#                 env = SquaresEnv()
#
#                 model = PPO('MlpPolicy', env, learning_rate=lr, batch_size=batch, n_epochs=10,
#                             gamma=0.99, gae_lambda=0.95, clip_range=clip, clip_range_vf=None, normalize_advantage=True,
#                             ent_coef=ent, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=- 1,
#                             target_kl=None, tensorboard_log=logdir, create_eval_env=False, policy_kwargs=None,
#                             verbose=1, seed=None, device='auto', _init_setup_model=True)
#
#                 pretrain_agent(model, env, train_expert_dataset, test_expert_dataset)
#                 TIMESTEPS = 10000000
#                 model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
#                 model_name = f"{models_dir}/{TIMESTEPS*random.randint(0,1000)}"
#                 model.save(model_name)


