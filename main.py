import random

import numpy as np
import pygame
import sys
import gym
from environment import SquaresEnv
from SquareGameRender import SquareGameRenderer
from pretrain import ExpertDataSet, pretrain_agent
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.data.dataset import random_split
from stable_baselines3 import PPO
import os
import time
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
check_env(myEnv)
#check_runtime(myEnv)

# PRETRAIN SECTION
os.chdir('expert')
files = os.listdir('.')
expert_observations = []
expert_actions = []
for file in files:
    data = np.load(file)
    expert_observations.append(data['expert_observations'][:])
    expert_actions.append(data['expert_actions'][:])
os.chdir('../')

obs = expert_observations[0]
acts = expert_actions[0]
for a in expert_observations:
    obs = np.concatenate((obs, a), axis=0)

for b in expert_actions:
    acts = np.concatenate((acts, b), axis=0)

expert_dataset = ExpertDataSet(obs, acts)

train_size = int(0.8 * len(expert_dataset))

test_size = len(expert_dataset) - train_size

train_expert_dataset, test_expert_dataset = random_split(
    expert_dataset, [train_size, test_size]
)
print("test_expert_dataset: ", len(test_expert_dataset))
print("train_expert_dataset: ", len(train_expert_dataset))
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

for lr in [0.003]:
    for batch in [128]:
        for clip in [0.4]:
            for ent in [0.01]:
                envType = 'linear_obs_action_MD_'
                run = envType + f'lr_{lr}_batch_{batch}_clip_{clip}_ent_{ent}_{np.random.randint(0, 1000)}'
                models_dir = f"models/{run}/"
                logdir = f"logs/{run}/"
                # Parallel environments
                #env = make_vec_env(SquaresEnv, n_envs=6)
                env = SquaresEnv()

                model = PPO('MlpPolicy', env, learning_rate=lr, batch_size=batch, n_epochs=10,
                            gamma=0.99, gae_lambda=0.95, clip_range=clip, clip_range_vf=None, normalize_advantage=True,
                            ent_coef=ent, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=- 1,
                            target_kl=None, tensorboard_log=logdir, create_eval_env=False, policy_kwargs=None,
                            verbose=1, seed=None, device='auto', _init_setup_model=True)

                mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
                print(f"Mean reward = {mean_reward} +/- {std_reward}")
                pretrain_agent(model, env, train_expert_dataset, test_expert_dataset)
                mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
                print(f"Mean reward = {mean_reward} +/- {std_reward}")

                TIMESTEPS = 5000000
                model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
                model_name = f"{models_dir}/{TIMESTEPS*random.randint(0,1000)}"
                model.save(model_name)


