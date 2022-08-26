from datetime import datetime
from blitting import LivePlot
from cartpoleEnv import CartPoleEnv
from tinyekf import EKF
import numpy as np

env = CartPoleEnv(render_mode='human', noise_lvl=0.05)
state = env.reset()
action = 0

lp = LivePlot(
    labels=(('x', 'x est'), ('theta', 'theta obs', 'theta est')),
    ymins=[-2.4, -np.pi/4],
    ymaxes=[2.4, np.pi/4],
)

class PitchEKF(EKF):
    def __init__(self):
        # 4 states, 1 observable
        EKF.__init__(self, n=4, m=1, pval=0.1, qval=0.001, rval=0.1)

    def f(self, state, action):
        x, x_dot, theta, theta_dot = state
        force = action

        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = masspole + masscart
        length = 0.5  # actually half the pole's length
        polemass_length = masspole * length
        dt = 1/30.

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (
            force + polemass_length * theta_dot**2 * sintheta
        ) / total_mass
        thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
        )
        xacc = temp - polemass_length * thetaacc * costheta / total_mass

        x = x + dt * x_dot
        x_dot = x_dot + dt * xacc
        theta = theta + dt * theta_dot
        theta_dot = theta_dot + dt * thetaacc

        new_state = np.array([x, x_dot, theta, theta_dot])
        state_dot = np.array([x_dot, xacc, theta_dot, thetaacc])

        return new_state, state_dot

    def h(self, obs):
        #imu will measure pitch (theta), at the 2nd index
            
        H = np.array([0, 0, 1, 0])
        h = np.dot(H, obs)

        return h, H

pitchEKF = PitchEKF()

pitchEKF.x = state


while True:

    #x, xdot, theta, thetadot
    K = [0.5, 0.1, 25, 1]

    obs, reward, done, state = env.step(action)

    x_obs, xdot_obs, theta_obs, thetadot_obs = obs
    x, xdot, theta, thetadot = state

    state_est = pitchEKF.step(theta_obs, action)
    x_est = state_est[0]
    theta_est = state_est[2]

    action = K @ np.array([x_obs, xdot_obs, theta, thetadot_obs])

    lp.plot(x, x_est, theta, theta_obs, theta_est)

    env.render()
    if done == True:
        state = env.reset()
        pitchEKF.reset(state)
        action=0