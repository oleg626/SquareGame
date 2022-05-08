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

for lr in [0.005]:
    for batch in [128]:
        for clip in [0.4]:
            for ent in [0.05]:
                envType = 't_obs_board_numShape37_action_MD_'
                run = envType + f'lr_{lr}_batch_{batch}_clip_{clip}_ent_{ent}'
                models_dir = f"models/{run}/"
                logdir = f"logs/{run}/"
                # Parallel environments
                env = make_vec_env(SquaresEnv, n_envs=8)

                model = PPO('MultiInputPolicy', env, learning_rate=lr, batch_size=batch, n_epochs=10,
                            gamma=0.99, gae_lambda=0.95, clip_range=clip, clip_range_vf=None, normalize_advantage=True,
                            ent_coef=ent, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=- 1,
                            target_kl=None, tensorboard_log=logdir, create_eval_env=False, policy_kwargs=None,
                            verbose=1, seed=None, device='auto', _init_setup_model=True)

                TIMESTEPS = 5000000

                model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
                model.save(f"{models_dir}/{TIMESTEPS*random.randint(0,1000)}")
