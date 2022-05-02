import gym
from gym import Env
from gym.spaces import Discrete, Box
import random
import numpy as np


def get_shape(shape_type):
    shape = np.zeros((2, 1))
    if shape_type == 1:  # X
        shape = np.ones((1, 1), dtype=np.uint8)
    elif shape_type == 2:  # XX
        shape = np.ones((2, 1), dtype=np.uint8)
    elif shape_type == 3:
        shape = np.ones((1, 2), dtype=np.uint8)
    elif shape_type == 4:  # XXX
        shape = np.ones((3, 1), dtype=np.uint8)
    elif shape_type == 5:
        shape = np.ones((1, 3), dtype=np.uint8)
    elif shape_type == 6:  # XXXX
        shape = np.ones((4, 1), dtype=np.uint8)
    elif shape_type == 7:
        shape = np.ones((1, 4), dtype=np.uint8)
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
        self.NUM_OF_SHAPES = 37
        self.NUM_OF_OPTIONS = 3
        self.BOARD_WIDTH = 9
        self.BOARD_HEIGHT = 9
        self.BOX_WIDTH = 3
        self.BOX_HEIGHT = 3

        self.action_space = Box(np.array([0, 0, 0]),
                                np.array([self.NUM_OF_OPTIONS - 1, self.BOARD_HEIGHT - 1, self.BOARD_WIDTH - 1]),
                                shape=(3, ), dtype=np.uint8)
        low = np.zeros(self.BOARD_HEIGHT * self.BOARD_WIDTH + self.NUM_OF_OPTIONS)
        # board part
        high1 = np.ones((self.BOARD_HEIGHT * self.BOARD_WIDTH, ), dtype=np.uint8)
        # shapes part
        high2 = np.full(self.NUM_OF_OPTIONS, self.NUM_OF_SHAPES)
        high = np.concatenate([high1, high2])
        self.observation_space = Box(low, high, shape=high.shape, dtype=np.uint8)

        self.board = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=np.uint8)
        self.items = np.zeros(self.NUM_OF_OPTIONS, dtype=np.uint8)
        self.items = random.random.randint(1, self.NUM_OF_SHAPES, (self.NUM_OF_OPTIONS,))

    def check_full(self):
        reward = 0
        bingo = np.ones((self.BOX_HEIGHT, self.BOX_WIDTH), dtype=np.uint8)
        for row in range(0, self.BOARD_HEIGHT - 1, self.BOX_HEIGHT):
            for col in range(0, self.BOARD_WIDTH - 1, self.BOX_WIDTH):
                if np.array_equal(self.board[row:row + self.BOX_HEIGHT, col:col+self.BOX_WIDTH], bingo):
                    self.board[row:row + self.BOX_HEIGHT, col:col+self.BOX_WIDTH] = 0
                    print('yay')
                    reward += 10
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
        for shape_type in self.items:
            if shape_type != 0:
                shape = get_shape(shape_type)
                for y in range(0, self.board.shape[0] - shape.shape[0]):
                    for x in range(0, self.board.shape[1] - shape.shape[1]):
                        if self.insertionPossible(shape, y, x):
                            there_are_options = True
        return there_are_options

    def step(self, action):
        self.counter += 1
        reward = 0
        done = False

        shape_index = action[0]
        y = action[1]
        x = action[2]

        # if chosen shape is not yet used
        if self.items[shape_index] != 0:
            shape_type = self.items[shape_index]
            self.items[shape_index] = 0
            shape = self.getShape(shape_type)
            shape_y = shape.shape[0]
            shape_x = shape.shape[1]

            if self.insertion_possible(shape, y, x):
                reward += 3
                self.board[y : y + shape_y, x : x + shape_x] = shape
            else:
                reward -= 3
        else:
            reward -= 3

        # add reward for bingo
        reward += self.checkFull()

        # update shapes available
        if np.array_equal(self.items, np.zeros((self.NUM_OF_OPTIONS,))):
            self.items = random.random.randint(1, self.NUM_OF_SHAPES, (self.NUM_OF_OPTIONS,))

        # check there are options
        done = not self.there_are_options()
        if done:
            print('shit')
        info = {}
        flat_state = np.concatenate(self.board.flatten(), self.items)
        print(flat_state)
        return flat_state, reward, done, info

    def reset(self):
        self.board = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=np.uint8)
        self.items = np.zeros(self.NUM_OF_OPTIONS, dtype=np.uint8)
        self.items = random.random.randint(1, self.NUM_OF_SHAPES, (self.NUM_OF_OPTIONS,))
        flat_state = np.concatenate(self.board.flatten(), self.items)
        return flat_state

    def render(self, mode='shit'):
        pass