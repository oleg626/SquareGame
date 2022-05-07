import random

import numpy as np
import pygame
import sys
import gym
from environment import SquaresEnv
from stable_baselines3 import PPO
import os
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
# It will check your custom environment and output additional warnings if needed

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


myEnv = SquaresEnv()
check_env(myEnv)
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"
# Parallel environments
env = make_vec_env(SquaresEnv, n_envs=6)
from stable_baselines3.common.envs import SimpleMultiObsEnv



model = PPO('MlpPolicy', env, learning_rate=0.01, batch_size=512, n_epochs=10,
            gamma=0.98, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, normalize_advantage=True,
            ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=- 1,
            target_kl=None, tensorboard_log=logdir, create_eval_env=False, policy_kwargs=None,
            verbose=1, seed=None, device='cuda', _init_setup_model=True)

TIMESTEPS = 5000000

model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
model.save(f"{models_dir}/{TIMESTEPS*random.randint(0,1000)}")

done = False
total_reward = 0
list_of_rewards = []
for i in range(0, 5):
    total_reward = 0
    done = False
    state = myEnv.reset()
    while not done:
        myEnv.render()
        actions = model.predict(state)
        print(actions[0])
        state, reward, done, info = myEnv.step(actions[0])
        total_reward += reward
    print(total_reward)
