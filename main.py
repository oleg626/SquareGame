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

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"
# Parallel environments
env = make_vec_env(SquaresEnv, n_envs=6)

from stable_baselines3.common.envs import SimpleMultiObsEnv



model = PPO('MultiInputPolicy', env, learning_rate=0.003, batch_size=64, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, clip_range_vf=None, normalize_advantage=True,
            ent_coef=0.05, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=- 1,
            target_kl=None, tensorboard_log=logdir, create_eval_env=False, policy_kwargs=None,
            verbose=1, seed=None, device='cuda', _init_setup_model=True)

TIMESTEPS = 1000000

model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
model.save(f"{models_dir}/{TIMESTEPS*random.randint(0,1000)}")

env = SquaresEnv()
done = False
total_reward = 0
list_of_rewards = []
for i in range(0, 5):
    total_reward = 0
    done = False
    state = env.reset()
    while not done:
        actions = model.predict(state)
        print(actions[0])
        state, reward, done, info = env.step(actions[0])
        env.render()
        total_reward += reward
    print(total_reward)
