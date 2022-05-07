import gym
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import time


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

    res = np.zeros((4, 4), dtype=np.uint8)
    res[:shape.shape[0], :shape.shape[1]] = shape
    return res


for i in range(1, 37):
    s = get_shape(i)


class SquaresEnv(Env):
    def __init__(self):
        self.total_reward = 0
        self.steps_made = 0
        self.NUM_OF_SHAPES = 37 # 3 actually
        self.BOARD_WIDTH = 9
        self.BOARD_HEIGHT = 9
        self.BOX_WIDTH = 3
        self.BOX_HEIGHT = 3

        self.action_space = gym.spaces.MultiDiscrete(([self.BOARD_HEIGHT, self.BOARD_WIDTH]))
        self.observation_space = Box(low=0, high=255,
                                     shape=(self.BOARD_HEIGHT*2 + 4, self.BOARD_WIDTH, 1), dtype=np.uint8)
        self.board = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=np.uint8)
        self.options = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=np.uint8)
        self.shape = get_shape(np.random.randint(1, self.NUM_OF_SHAPES))

    def check_full(self):
        reward = 0
        bingo = np.ones((self.BOX_HEIGHT, self.BOX_WIDTH), dtype="float32")
        for row in range(0, self.BOARD_HEIGHT - 1, self.BOX_HEIGHT):
            for col in range(0, self.BOARD_WIDTH - 1, self.BOX_WIDTH):
                if np.array_equal(self.board[row:row + self.BOX_HEIGHT, col:col+self.BOX_WIDTH], bingo):
                    self.board[row:row + self.BOX_HEIGHT, col:col+self.BOX_WIDTH] = 0
                    #print('yay')
                    reward += 1
        return reward

    def insertion_possible(self, shape, y, x, log = False):
        shape_y = shape.shape[0]
        shape_x = shape.shape[1]
        if (y + shape_y) <= self.board.shape[0] and (x + shape_x) <= self.board.shape[1]:
            # check collision
            for local_x in range(0, shape_x):
                for local_y in range(0, shape_y):
                    if self.board[y + local_y, x + local_x] == 1 and shape[local_y, local_x] == 1:
                        if log:
                            print(f'board {y + local_y} , {x+ local_x} , shape {local_y}, {local_x}')
                        return False
        else:
            if log:
                print("size issue")
            return False
        return True

    def there_are_options(self, log = False):
        options = np.zeros(shape=(self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype="float32")
        shape = self.shape
        p = np.where(shape != 0)
        shape = shape[min(p[0]): max(p[0]) + 1, min(p[1]): max(p[1]) + 1]
        if log:
            print("there are options, shape")
            print(shape)
        for y in range(0, self.board.shape[0] - shape.shape[0] + 1):
            for x in range(0, self.board.shape[1] - shape.shape[1] + 1):
                if log:
                    print(f'check at pos ({y},{x})')
                if self.insertion_possible(shape, y, x, log):
                    if log:
                        print('passed')
                    options[y, x] = 1
        return options

    def step(self, action):
        self.steps_made += 1
        reward = 0
        done = False
        y = action[0]
        x = action[1]
        shape = self.shape
        p = np.where(shape != 0)
        shape = shape[min(p[0]): max(p[0]) + 1, min(p[1]): max(p[1]) + 1]
        shape_y = shape.shape[0]
        shape_x = shape.shape[1]

        if self.insertion_possible(shape, y, x):
            reward += 0.2
            self.board[y : y + shape_y, x : x + shape_x] = np.add(self.board[y : y + shape_y, x : x + shape_x], shape)
        else:
            reward -= 1
            #done = True
            pass
        # add reward for bingo
        reward += self.check_full()
        # update shapes available
        self.shape = get_shape(np.random.randint(1, self.NUM_OF_SHAPES))

        # check there are options
        self.total_reward += reward
        self.options = self.there_are_options()
        done = done or np.array_equal(self.options, np.zeros(shape=(self.BOARD_HEIGHT, self.BOARD_WIDTH)))
        done = done or self.steps_made >= 100

        info = {}
        state = np.zeros(shape=(self.BOARD_HEIGHT*2 + 4, self.BOARD_WIDTH, 1), dtype=np.uint8)
        state[:self.BOARD_HEIGHT, :, 0] = self.board
        state[self.BOARD_HEIGHT: 2*self.BOARD_HEIGHT, :, 0] = self.options
        state[self.BOARD_HEIGHT*2:, : self.shape.shape[1], 0] = self.shape
        state *= 255
        return state, reward, done, info

    def reset(self):
        self.steps_made = 0
        self.total_reward = 0
        self.board = np.zeros((self.BOARD_HEIGHT, self.BOARD_WIDTH), dtype=np.uint8)
        self.shape = get_shape(np.random.randint(1, self.NUM_OF_SHAPES))
        state = np.zeros(shape=(self.BOARD_HEIGHT*2 + 4, self.BOARD_WIDTH, 1), dtype=np.uint8)
        state[:self.BOARD_HEIGHT, :, 0] = self.board
        state[self.BOARD_HEIGHT: 2*self.BOARD_HEIGHT, :, 0] = self.options
        state[self.BOARD_HEIGHT*2:, : self.shape.shape[1], 0] = self.shape
        state *= 255
        return state

    def render(self, mode='shit'):
        print(self.board)
        print(self.shape)
        print(self.total_reward)
        print('')
        pass