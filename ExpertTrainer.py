from SquareGameRender import SquareGameRenderer
from environment import SquaresEnv

myEnv = SquaresEnv()
state = myEnv.reset()
game = SquareGameRenderer(myEnv, myEnv.get_obs(), myEnv.get_shape(), num_episodes=20)
game.start()
