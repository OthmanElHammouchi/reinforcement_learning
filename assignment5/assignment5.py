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

rand_env = gym.make("FrozenLake8x8-v1", desc = None, map_name = None, is_slippery=True).env

alpha = 0.999

# decay_schedule = lambda t: poly_schedule(t, 0.51)
decay_schedule = KSchedule(0.51, 5000)

Q, policy = learn_policy(rand_env, alpha, decay_schedule, n_runs = int(1e5), q_init=0)

sim_policy(rand_env, policy)

eval_policy(rand_env, policy)
