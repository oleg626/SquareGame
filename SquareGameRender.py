import turtle
import numpy as np
from pretrain import pretrain_agent


class SquareGameRenderer:
    def __init__(self, env, state_shape, figure_shape, num_episodes=1):
        self.env = env
        self.num_episodes = num_episodes
        self.current_episode = 0
        self.window = turtle.Screen()
        self.window.title("SquareGame")
        self.board_width = state_shape[1]
        self.board_height = state_shape[0]
        self.shape_width = figure_shape[1]
        self.shape_height = figure_shape[0]
        self.general_margin_x = 60
        self.general_margin_y = 60
        self.icon_width = 100
        self.icon_height = 100
        self.window_width = (self.board_width + self.shape_width) * self.icon_width + 200
        self.window_height = self.board_height * self.icon_height + 200
        self.window.setup(self.window_width, self.window_height, startx = None, starty = None)
        self.half_window_width = self.window_width / 2
        self.half_window_height = self.window_height / 2
        self.window.bgcolor("black")
        self.window.register_shape("cookie.gif")
        self.window.register_shape("cupcake.gif")
        self.window.register_shape("save.gif")
        self.window.delay(0)
        self.state = []

        self.expert_observations = []
        self.expert_actions = []

    def step(self, x, y):
        x += (self.half_window_width - self.general_margin_x)
        x = round(x / self.icon_width)
        y += (self.half_window_height + self.general_margin_y)
        y = abs(self.window_height - y)
        y = round(y / self.icon_height)
        if x >= self.board_width or y >= self.board_height:
            return

        action = [y, x]

        self.expert_observations.append(self.state)
        self.expert_actions.append(action)

        self.state, reward, done, info = self.env.step(action)
        total_reward = info['total_reward']
        if done:
            self.state = self.env.reset()
            total_reward = 0
            self.current_episode += 1
            if self.current_episode == self.num_episodes:
                np.savez_compressed(
                    f"expert/expert_data_{np.random.randint(1, 100000)}",
                    expert_actions=self.expert_actions,
                    expert_observations=self.expert_observations,
                )
                print("Expert data saved")
        self.redraw(self.state, total_reward)

    def start(self):
        self.state = self.env.reset()
        self.redraw(self.state, 0)
        self.window.onscreenclick(self.step)
        self.window.mainloop()

    def save_data(self, x, y):
        np.savez_compressed(
            f"expert/expert_data_{np.random.randint(1, 100000)}",
            expert_actions=self.expert_actions,
            expert_observations=self.expert_observations,
        )
        print('data saved')

    def redraw(self, state, reward):
        self.window.clearscreen()
        self.window.bgcolor("black")
        self.window.delay(0)
        self.window.onscreenclick(self.step)

        save_icon = turtle.Turtle()
        save_icon.shape("save.gif")
        save_icon.penup()
        save_icon.speed(0)
        save_icon.setposition(500, -300)
        save_icon.onclick(self.save_data)

        idx = self.board_height * self.board_width
        board = np.array(state[:idx])
        board = board.reshape((self.board_width, self.board_height))
        shape = np.array(state[idx:])
        shape = shape.reshape((self.shape_width, self.shape_height))
        reward_text = turtle.Turtle()
        reward_text.hideturtle()
        reward_text.color("white")
        reward_text.penup()
        reward_text.setposition(-self.half_window_width + 300, -self.half_window_height + 50)
        reward_text.write(f"Reward: {round(reward, 1)}", align="center", font=("Courier New", 32, "normal"))
        for row in range(self.board_height):
            for col in range(self.board_width):
                cookie = turtle.Turtle()
                cookie.penup()
                cookie.speed(0)
                if board[row, col] == 0:
                    cookie.shape("cupcake.gif")
                else:
                    cookie.shape("cookie.gif")
                x_pos = self.general_margin_x + col * self.icon_width - self.half_window_width
                y_pos = self.window_height - (row * self.icon_height) - self.half_window_height - self.general_margin_y
                cookie.setposition(x_pos, y_pos)
        shape_margin_x = (self.general_margin_x * 2) + board.shape[1] * self.icon_width
        for row in range(self.shape_height):
            for col in range(self.shape_width):
                cookie = turtle.Turtle()
                cookie.penup()
                cookie.speed(0)
                if shape[row, col] == 1:
                    cookie.shape("cookie.gif")
                else:
                    cookie.shape("cupcake.gif")
                x_pos = shape_margin_x + col * self.icon_width - self.half_window_width
                y_pos = self.window_height - ((row + 1) * self.icon_height) - self.half_window_height
                cookie.setposition(x_pos, y_pos)

