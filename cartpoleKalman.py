from datetime import datetime
from blitting import LivePlot
from cartpoleEnv import CartPoleEnv
from tinyekf import EKF
import numpy as np

env = CartPoleEnv(render_mode='human', noise_lvl=0.1)
obs = env.reset()
action = 0

lp = LivePlot(
    labels=('x', ('theta', 'theta obs', 'theta est')),
    ymins=[-4, -np.pi/4],
    ymaxes=[4, np.pi/4],
)

class PitchEKF(EKF):
    def __init__(self):
        # 4 states, 1 observable
        EKF.__init__(self, n=4, m=1, rval=0.001)

    def f(self, x):

        g = 9.8
        L = 0.5

        F = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, g/L, 0],            
        ])
        F = F@x*1/30.

        x_new = x + F@x
        
        return x_new, F

    def h(self, obs):
        #imu will measure pitch (theta), at the 2nd index
            
        H = np.array([0, 0, 1, 0])
        h = np.dot(H, obs)

        return h, H

pitchEKF = PitchEKF()


while True:

    #x, xdot, theta, thetadot
    K = [1, 0, 20, 0]

    obs, reward, done, state = env.step(action)

    x_obs, xdot_obs, theta_obs, thetadot_obs = obs
    x, xdot, theta, thetadot = state

    theta_est = pitchEKF.step(theta_obs)[2]

    action = K @ np.array([x_obs, xdot_obs, theta_obs, thetadot_obs])

    lp.plot(x, theta, theta_obs, theta_est)

    env.render()
    if done == True:
        env.reset()
        pitchEKF.x = np.array([0., 0, 0, 0])