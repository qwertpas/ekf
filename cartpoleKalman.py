from datetime import datetime
from blitting import LivePlot
from cartpoleEnv import CartPoleEnv
from tinyekf import EKF
import numpy as np

env = CartPoleEnv(render_mode='human', noise_lvl=0.1)
state = env.reset()
action = 0

lp = LivePlot(
    labels=(
        ('x_obs', 'x_est'), 
        ('theta_obs', 'theta_est'),
        ('xdot_obs', 'xdot_est'), 
        ('thetadot_obs', 'thetadot_est'),
        ('error', '0')
    ),
    ymins=[-2.4, -np.pi/4, -1, -1, 0],
    ymaxes=[2.4, np.pi/4, 1, 1, 0.5],
)

class PitchEKF(EKF):
    def __init__(self):
        # 4 states, 1 observable
        EKF.__init__(self, n=4, m=4, pval=0.1, qval=0.25, rval=2)

    def f(self, state, action):
        x, x_dot, theta, theta_dot = state
        force = action

        gravity = 9.8
        masscart = 1.0
        masspole = 0.15
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

        return new_state, np.diag(state_dot)

    def h(self, obs):
        #imu will measure pitch (theta), at the 2nd index
            
        # H = np.array([1, 1, 1, 1])
        H = np.eye(4)
        h = np.dot(H, obs)

        return h, H

pitchEKF = PitchEKF()

pitchEKF.x = state

errors = np.zeros(10)


log = {
    'timestamp': [],
    'x_obs': [],
    'xdot_obs': [],
    'theta_obs': [],
    'thetadot_obs': [],
    'x': [],
    'xdot': [],
    'theta': [],
    'thetadot': [],
    'x_est': [],
    'xdot_est': [],
    'theta_est': [],
    'thetadot_est': [],
    'action': [],
}


while len(log['timestamp']) < 100:

    #x, xdot, theta, thetadot
    K = [0.5, 0.3, 20, 1]

    obs, reward, done, state = env.step(action)

    x_obs, xdot_obs, theta_obs, thetadot_obs = obs
    x, xdot, theta, thetadot = state

    state_est = pitchEKF.step(obs, action)

    errors = np.roll(errors, -1)
    errors[-1] = np.linalg.norm(state_est - state)
    # print(errors)
    # print(np.mean(errors))

    x_est, xdot_est, theta_est, thetadot_est = state_est

    # action = K @ np.array([x_est, xdot_est, theta_est, thetadot_est])
    action = K @ state

    log['timestamp'].append(datetime.now().timestamp())

    log['x_obs'].append(x_obs)
    log['xdot_obs'].append(xdot_obs)
    log['theta_obs'].append(theta_obs)
    log['thetadot_obs'].append(thetadot_obs)

    log['x'].append(x)
    log['xdot'].append(xdot)
    log['theta'].append(theta)
    log['thetadot'].append(thetadot)

    log['x_est'].append(x_est)
    log['xdot_est'].append(xdot_est)
    log['theta_est'].append(theta_est)
    log['thetadot_est'].append(thetadot_est)

    log['action'].append(action)


    lp.plot(
        x_obs, x_est, 
        theta_obs, theta_est,
        xdot_obs, xdot_est,
        thetadot_obs, thetadot_est,
        np.mean(errors), 0
    )

    env.render()
    if done == True:
        state = env.reset()
        pitchEKF.reset(state)
        action=0

import pandas as pd
pd.DataFrame.from_dict(log).to_csv("cartpole_log_n0.1.csv", index=False)