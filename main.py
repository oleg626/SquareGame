import numpy as np
import pygame
import sys
import gym
from environment import SquaresEnv
from stable_baselines3 import PPO
import os
import time
from stable_baselines3.common.env_util import make_vec_env

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"
# Get the environment and extract the number of actions.
# Parallel environments
env = make_vec_env(SquaresEnv, n_envs=10)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 1000000
iters = 10000000
while iters > 0:
	iters -= 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")