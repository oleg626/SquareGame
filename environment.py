import gym
from gym import Env
from gym.spaces import Discrete, Box
import random
import numpy as np
import pygame
import cv2
import sys


def get_shape(shape_type):
    shape = np.zeros((2, 1))
    if shape_type == 1:  # X
        shape = np.ones((1, 1), dtype=np.uint8)
    elif shape_type == 2:  # XX
        shape = np.ones((1, 2), dtype=np.uint8)
    elif shape_type == 3:
        shape = np.ones((2, 1), dtype=np.uint8)
    elif shape_type == 4:  # XXX
        shape = np.ones((1, 3), dtype=np.uint8)
    elif shape_type == 5:
        shape = np.ones((3, 1), dtype=np.uint8)
    elif shape_type == 6:  # XXXX
        shape = np.ones((1, 4), dtype=np.uint8)
    elif shape_type == 7:
        shape = np.ones((4, 1), dtype=np.uint8)
    elif shape_type == 8:  # square corner
        shape = np.ones((2, 2), dtype=np.uint8)
        shape[0, 0] = 0
    elif shape_type == 9:
        shape = np.ones((2, 2), dtype=np.uint8)
        shape[0, 1] = 0
    elif shape_type == 10:
        shape = np.ones((2, 2), dtype=np.uint8)
        shape[1, 0] = 0
    elif shape_type == 11:
        shape = np.ones((2, 2), dtype=np.uint8)
        shape[1, 1] = 0
    elif shape_type == 12:  # square one sided long vertical
        shape = np.ones((3, 2), dtype=np.uint8)
        shape[1:, 1] = 0
    elif shape_type == 13:
        shape = np.ones((3, 2), dtype=np.uint8)
        shape[1:, 0] = 0
    elif shape_type == 14:
        shape = np.ones((3, 2), dtype=np.uint8)
        shape[:2, 1] = 0
    elif shape_type == 15:
        shape = np.ones((3, 2), dtype=np.uint8)
        shape[:2, 0] = 0
    elif shape_type == 16:  # square one sided long horizontal
        shape = np.ones((2, 3), dtype=np.uint8)
        shape[0, 1:] = 0
    elif shape_type == 17:
        shape = np.ones((2, 3), dtype=np.uint8)
        shape[0, :2] = 0
    elif shape_type == 18:
        shape = np.ones((2, 3), dtype=np.uint8)
        shape[1, 1:] = 0
    elif shape_type == 19:
        shape = np.ones((2, 3), dtype=np.uint8)
        shape[1, :2] = 0

    elif shape_type == 20:  # straight big corner
        shape = np.ones((3, 3), dtype=np.uint8)
        shape[:2, 1:] = 0
    elif shape_type == 21:
        shape = np.ones((3, 3), dtype=np.uint8)
        shape[:2, :2] = 0
    elif shape_type == 22:
        shape = np.ones((3, 3), dtype=np.uint8)
        shape[1:, 1:] = 0
    elif shape_type == 23:
        shape = np.ones((3, 3), dtype=np.uint8)
        shape[1:, :2] = 0
    elif shape_type == 24:  # T shape
        shape = np.ones((3, 3), dtype=np.uint8)
        shape[0, 1:] = 0
        shape[2, 1:] = 0
    elif shape_type == 25:
        shape = np.ones((3, 3), dtype=np.uint8)
        shape[:2, 0] = 0
        shape[:2, 2] = 0
    elif shape_type == 26:
        shape = np.ones((3, 3), dtype=np.uint8)
        shape[0, :2] = 0
        shape[2, :2] = 0
    elif shape_type == 27:
        shape = np.ones((3, 3), dtype=np.uint8)
        shape[1:, 0] = 0
        shape[1:, 2] = 0
    elif shape_type == 28:  # short T shape
        shape = np.ones((3, 2), dtype=np.uint8)
        shape[0, 1] = 0
        shape[2, 1] = 0
    elif shape_type == 29:
        shape = np.ones((3, 2), dtype=np.uint8)
        shape[0, 0] = 0
        shape[2, 0] = 0
    elif shape_type == 30:
        shape = np.ones((2, 3), dtype=np.uint8)
        shape[1, 0] = 0
        shape[1, 2] = 0
    elif shape_type == 31:
        shape = np.ones((2, 3), dtype=np.uint8)
        shape[0, 0] = 0
        shape[0, 2] = 0
    elif shape_type == 32:  # S shape
        shape = np.ones((2, 3), dtype=np.uint8)
        shape[0, 0] = 0
        shape[1, 2] = 0
    elif shape_type == 33:
        shape = np.ones((2, 3), dtype=np.uint8)
        shape[1, 0] = 0
        shape[0, 2] = 0
    elif shape_type == 34:
        shape = np.ones((3, 2), dtype=np.uint8)
        shape[0, 1] = 0
        shape[2, 0] = 0
    elif shape_type == 35:
        shape = np.ones((3, 2), dtype=np.uint8)
        shape[0, 0] = 0
        shape[2, 1] = 0
    elif shape_type == 36:  # square
        shape = np.ones((2, 2), dtype=np.uint8)
    elif shape_type == 37:  # cross
        shape = np.zeros((3, 3), dtype=np.uint8)
        shape[:, 1] = 1
        shape[1, :] = 1
    return shape


class SquaresEnv(Env):
    def __init__(self):
        self.total_reward = 0
        self.NUM_OF_SHAPES = 8 # 3 actually
        self.NUM_OF_OPTIONS = 3
        self.BOARD_WIDTH = 6
        self.BOARD_HEIGHT = 6
        self.BOX_WIDTH = 3
        self.BOX_HEIGHT = 3

        self.action_space = Box(low=0, high=0, shape=((self.BOARD_HEIGHT + self.BOARD_WIDTH + self.NUM_OF_SHAPES),),
                                dtype=np.uint8)
        self.observation_space = Box(low=0, high=1, shape=((self.BOARD_HEIGHT * self.BOARD_WIDTH + self.NUM_OF_SHAPES),),
                                     dtype=np.uint8)

        self.board = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=np.uint8)
        self.items = np.zeros(self.NUM_OF_SHAPES, dtype=np.uint8)
        random_items = np.random.randint(1, self.NUM_OF_SHAPES, (self.NUM_OF_OPTIONS,))
        for i in random_items:
            self.items[i] = 1

    def check_full(self):
        reward = 0
        bingo = np.ones((self.BOX_HEIGHT, self.BOX_WIDTH), dtype=np.uint8)
        for row in range(0, self.BOARD_HEIGHT - 1, self.BOX_HEIGHT):
            for col in range(0, self.BOARD_WIDTH - 1, self.BOX_WIDTH):
                if np.array_equal(self.board[row:row + self.BOX_HEIGHT, col:col+self.BOX_WIDTH], bingo):
                    self.board[row:row + self.BOX_HEIGHT, col:col+self.BOX_WIDTH] = 0
                    print('yay')
                    reward += 5
        return reward

    def insertion_possible(self, shape, y, x):
        shape_y = shape.shape[0]
        shape_x = shape.shape[1]
        if (y + shape_y) <= self.board.shape[0] and (x + shape_x) <= self.board.shape[1]:
            # check collision
            for local_x in range(0, shape_x):
                for local_y in range(0, shape_y):
                    if self.board[y + local_y, x + local_x] == 1 and shape[local_y, local_x] == 1:
                        return False
        else:
            return False
        return True

    def there_are_options(self):
        there_are_options = False
        for shape_type in range(1, self.items):
            if self.items[shape_type] != 0:
                shape = get_shape(self.items[shape_type])
                for y in range(0, self.board.shape[0] - shape.shape[0]):
                    for x in range(0, self.board.shape[1] - shape.shape[1]):
                        if self.insertion_possible(shape, y, x):
                            there_are_options = True
        return there_are_options

    def step(self, action):
        reward = 0
        done = False
        #print(action)
        act_y = action[:self.BOARD_HEIGHT]
        act_x = action[self.BOARD_HEIGHT: self.BOARD_HEIGHT + self.BOARD_WIDTH + 1]
        action_shape = action[-self.NUM_OF_SHAPES:]
        print(act_x, act_y, action_shape)
        shape_index = action_shape.tolist().index(1)
        y = act_y.tolist().index(1)
        x = act_x.tolist().index(1)
        print (shape_index, y, x)
        # if chosen shape is not yet used
        if self.items[shape_index] != 0:
            shape_type = self.items[shape_index]
            shape = get_shape(shape_type)
            shape_y = shape.shape[0]
            shape_x = shape.shape[1]

            if self.insertion_possible(shape, y, x):
                self.items[shape_index] = 0
                reward += 3
                self.board[y : y + shape_y, x : x + shape_x] = np.add(self.board[y : y + shape_y, x : x + shape_x], shape)
            else:
                reward -= 1
        else:
            reward -= 1

        # add reward for bingo
        reward += self.check_full()

        # update shapes available
        if np.array_equal(self.items, np.zeros((self.NUM_OF_OPTIONS,))):
            self.items = np.random.randint(1, self.NUM_OF_SHAPES, (self.NUM_OF_OPTIONS,))

        # check there are options
        self.total_reward += reward
        done = not self.there_are_options() or reward > 6 or self.total_reward < -20
        # if done:
        #     print(self.total_reward)
        info = {}
        #print(self.board)
        # print(self.items)
        items = np.zeros(self.NUM_OF_SHAPES)
        for i in self.items:
            if i != 0:
                items[i] = 1
        flat_state = np.concatenate([self.board.flatten(), items])
        return flat_state, reward, done, info

    def reset(self):
        # print('reset')
        self.total_reward = 0
        self.board = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=np.uint8)
        self.items = np.zeros(self.NUM_OF_OPTIONS, dtype=np.uint8)
        self.items = np.random.randint(1, self.NUM_OF_SHAPES, (self.NUM_OF_OPTIONS,))
        flat_state = np.concatenate([self.board.flatten(), self.items])
        return flat_state

    def render(self, mode='shit'):
        # img = np.zeros((500, 500, 3), dtype = 'uint8')
        # cv2.imshow("game", img)
        # cv2.waitKey(0)

        pass