import cv2
import numpy as np


class SquareGameRenderOpenCv:
    def __init__(self, state_shape, figure_shape):
            self.current_episode = 0
            self.max_reward = 0
            self.board_width = state_shape[1]
            self.board_height = state_shape[0]
            self.shape_width = figure_shape[1]
            self.shape_height = figure_shape[0]
            self.general_margin_x = 100
            self.general_margin_y = 100
            self.icon_width = 64
            self.icon_height = 64
            self.window_width = (self.board_width + self.shape_width) * self.icon_width + 200
            self.window_height = self.board_height * self.icon_height + 200
            self.half_window_width = self.window_width / 2
            self.half_window_height = self.window_height / 2
            self.background = "bb.png"
            self.taken = "poppy_bun.gif"
            self.free = "octopus.gif"

    def render(self, state, figure, reward):
        window = cv2.imread(self.background)
        cv2.imshow("SquareGame", window)
        cv2.waitKey()
