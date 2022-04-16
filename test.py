from CartPoleContinuous import CartPoleContinuousEnv
import numpy as np

env = CartPoleContinuousEnv()
env.reset()

action = np.array([1.])
print(env.step([0.80]))