from datetime import datetime
from blitting import LivePlot
from cartpoleEnv import CartPoleEnv
from tinyekf import EKF
import numpy as np

env = CartPoleEnv(render_mode='human', noise_lvl=0.1)
obs = env.reset()

lp = LivePlot(
    labels=('x', ('theta', 'theta obs', 'theta est')),
    ymins=[-4, -np.pi/2],
    ymaxes=[4, np.pi/2],
)


while True:

    #x, xdot, theta, thetadot
    K = [1, 0, 20, 0]

    action = K @ obs
    obs, reward, done, state = env.step(action)

    x_obs, xdot_obs, theta_obs, thetadot_obs = obs
    x, xdot, theta, thetadot = state
    lp.plot(x, theta, theta_obs, theta_obs)

    # Render the game
    env.render()
    if done == True:
        env.reset()

    print(datetime.now())
