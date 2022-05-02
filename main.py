import numpy as np
import pygame
import sys
import gym
from environment import SquaresEnv

ENV_NAME = 'Pendulum-v1'



# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
myEnv = SquaresEnv()
print(myEnv.action_space)
print(myEnv.observation_space)

rand_int2 = np.random.randint(0,10,(3,)) # random numpy array of shape (4,5)

print (rand_int2)
print (np.array_equal(rand_int2, np.zeros((3,))))
#def run():