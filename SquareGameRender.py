import turtle
import numpy as np


class SquareGameRenderer:
    def __init__(self, env, state_shape, figure_shape, num_episodes=1):
        self.env = env
        self.num_episodes = num_episodes
        self.current_episode = 0
        self.max_reward = 0

        self.window = turtle.Screen()
        self.window.title("SquareGame")
        self.board_width = state_shape[1]
        self.board_height = state_shape[0]
        self.shape_width = figure_shape[1]
        self.shape_height = figure_shape[0]
        self.icon_width = 70
        self.icon_height = 70
        self.window_width = (self.board_width + self.shape_width + 2) * self.icon_width
        self.window_height = (self.board_height + 2) * self.icon_height
        self.window.setup(self.window_width, self.window_height, startx = None, starty = None)
        self.half_window_width = self.window_width / 2
        self.half_window_height = self.window_height / 2
        self.window.bgpic("bg.gif")
        self.taken = "poppy_bun.gif"
        self.free = "octopus.gif"
        self.window.register_shape(self.taken)
        self.window.register_shape(self.free)
        self.reward_text = turtle.Turtle()
        self.max_reward_text = turtle.Turtle()

        self.window.register_shape("save.gif")
        self.window.delay(0)
        self.state = []

        self.expert_observations = []
        self.expert_actions = []

        self.state_turtles = []
        for i in range(self.board_height):
            arr = []
            for j in range(self.board_width):
                arr.append(turtle.Turtle())
            self.state_turtles.append(arr)

        self.shape_turtles = []
        for i in range(self.shape_height):
            arr = []
            for j in range(self.shape_width):
                arr.append(turtle.Turtle())
            self.shape_turtles.append(arr)

    def step(self, x, y):
        x += (self.half_window_width - self.icon_width)
        x = round(x / self.icon_width)
        y += (self.half_window_height + self.icon_height)
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
            self.max_reward = max(self.max_reward, total_reward)
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
        for row in range(self.board_height):
            for col in range(self.board_width):
                cookie = turtle.Turtle()
                cookie.shape(self.free)
                cookie.penup()
                cookie.speed(0)
                x_pos = (col + 1) * self.icon_width - self.half_window_width
                y_pos = self.window_height - (row * self.icon_height) - self.half_window_height - self.icon_height
                cookie.setposition(x_pos, y_pos)
                self.state_turtles[row][col] = cookie

        shape_margin_x = (2 + self.board_width) * self.icon_width
        for row in range(self.shape_height):
            for col in range(self.shape_width):
                cookie = turtle.Turtle()
                cookie.shape(self.free)
                cookie.penup()
                cookie.speed(0)
                x_pos = shape_margin_x + col * self.icon_width - self.half_window_width
                y_pos = self.window_height - ((row + 1) * self.icon_height) - self.half_window_height
                cookie.setposition(x_pos, y_pos)
                self.shape_turtles[row][col] = cookie

        save_icon = turtle.Turtle()
        save_icon.shape("save.gif")
        save_icon.penup()
        save_icon.speed(0)
        save_icon.setposition(self.half_window_width - 130, -100)
        save_icon.onclick(self.save_data)

        self.reward_text.hideturtle()
        self.reward_text.color("white")
        self.reward_text.penup()
        self.reward_text.setposition(-self.half_window_width + 200, -self.half_window_height + 20)
        self.reward_text.write(f"Score: {round(0, 1)}", align="center", font=("Courier New", 28, "normal"))

        self.max_reward_text.hideturtle()
        self.max_reward_text.color("white")
        self.max_reward_text.penup()
        self.max_reward_text.setposition(self.half_window_width - 300, -self.half_window_height + 20)
        self.max_reward_text.write(f"Record: {round(self.max_reward, 1)}", align="center", font=("Courier New", 28, "normal"))

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
        idx = self.board_height * self.board_width
        board = np.array(state[:idx])
        board = board.reshape((self.board_width, self.board_height))
        shape = np.array(state[idx:])
        shape = shape.reshape((self.shape_width, self.shape_height))

        self.reward_text.clear()
        self.reward_text.write(f"Score: {round(reward, 1)}", align="center", font=("Courier New", 28, "normal"))
        self.max_reward_text.clear()
        self.max_reward_text.write(f"Record: {round(self.max_reward, 1)}", align="center",
                                   font=("Courier New", 28, "normal"))

        for row in range(self.board_height):
            for col in range(self.board_width):
                if board[row, col] == 0:
                    self.state_turtles[row][col].shape(self.free)
                else:
                    self.state_turtles[row][col].shape(self.taken)
        for row in range(self.shape_height):
            for col in range(self.shape_width):
                if shape[row, col] == 1:
                    self.shape_turtles[row][col].shape(self.taken)
                else:
                    self.shape_turtles[row][col].shape(self.free)
