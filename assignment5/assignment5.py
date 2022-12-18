import os; os.chdir("assignment5")
import numpy as np
import gym
import scipy.optimize as optimize
from q_learning import *

np.seterr('raise')

#################### deterministic environment ####################

# det_env = gym.make("FrozenLake8x8-v1", desc = None, map_name = None, is_slippery=False, render_mode = "ansi").env

# alpha = 0.999

# Q, policy = learn_policy(det_env, alpha, max_iter = int(1e6), q_init=10, lr_eps=0.01)
# # rand_policy = gen_rand_policy(rand_env)

# sim_policy(det_env, policy)

#################### stochastic environment ####################

rand_env = gym.make("FrozenLake8x8-v1", desc = None, map_name = None, is_slippery=True, render_mode = "ansi").env

alpha = 0.999

Q, policy = learn_policy(rand_env, alpha, n_runs = int(1e4), q_init=10, omega = 0.7, epsilon=0.5, K=5000)

sim_policy(rand_env, policy)

eval_policy(rand_env, policy)
