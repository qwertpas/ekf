from cartpoleEnv import CartPoleEnv
from tinyekf import EKF
import numpy as np

env = CartPoleEnv(render_mode='human')
obs = env.reset()


# class CartpoleEKF(EKF):

#     def __init__(self):

#         # One state (ASL), two measurements (baro, sonar), with larger-than-usual
#         # measurement covariance noise to help with sonar blips.
#         EKF.__init__(self, 1, 2, rval=.5)

#     def f(self, x):

#         # State-transition function is identity
#         return np.copy(x), np.eye(1)


#     def h(self, x):

#         # State value is ASL
#         asl = x[0]

#         # Convert ASL cm to sonar AGL cm by subtracting off ASL baseline from baro
#         s = sonarfun(asl - baro2asl(BARO_BASELINE))

#         # Convert ASL cm to Pascals: see http://www.engineeringtoolbox.com/air-altitude-pressure-d_462.html
#         b = asl2baro(asl)

#         h = np.array([b, s])

#         # First derivative of nonlinear baro-measurement function
#         # Used http://www.wolframalpha.com
#         dpdx = -0.120131 * pow((1 - 2.2577e-7 * x[0]), 4.25588)

#         # Sonar response is linear, so derivative is constant
#         dsdx = 0.933

#         H = np.array([[dpdx], [dsdx]])

#         return h, H


while True:

    #x, xdot, theta, thetadot
    K = [1, 0, 20, 0]

    action = K @ obs
    obs, reward, done = env.step(action)

    # Render the game
    env.render()
    if done == True:
        env.reset()
