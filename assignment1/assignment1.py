import gym
import numpy as np

# We will load a DiscreteEnv and retrieve the probability and reward
# information
env = gym.make("FrozenLake8x8-v1", desc=None, map_name=None)
"""
DiscreteEnv has an attribute P which holds everything er want as a
dictionary of lists:
P[s][a] = [(probability, nextstate, reward, done), ...]
"""
# first, we initialize the structures with zeros
prob = {i: {j: {a: 0 for a in range(env.nA)}
            for j in range(env.nS)}
        for i in range(env.nS)}
rewd = {i: {j: {a: 0 for a in range(env.nA)}
            for j in range(env.nS)}
        for i in range(env.nS)}
# then, we fill them with the actual information
for i in range(env.nS):
    for a in range(env.nA):
        for (p, j, r, d) in env.P[i][a]:
            prob[i][j][a] += p
            rewd[i][j][a] += r

# Policy computation: here's where YOU code
"""
Insert your clever policy computation here! make sure to replace the
policy dictionary below by the results of your computation
"""
T = int(1e3)
n_actions = env.nA
n_states = env.nS

P = np.zeros((n_actions, n_states, n_states))
R = np.zeros((n_actions, n_states, n_states))

for a in range(n_actions):
	for i in range(n_states):
		for (p, j, r, d) in env.P[i][a]:
			P[a, i, j] += p
			R[a, i, j] += r

def fh_policy(P, R, T):
	policy = np.empty((n_states, T), dtype = int)
	V = np.empty((n_states, T))

	for i in range(n_states):
		candidates = np.empty(n_actions)
		for a in range(n_actions):
			candidates[a] = np.dot(P[a, i, :], R[a, i, :])
		action = np.argmax(candidates)
		policy[i, T-1] = action
		V[i, T-1] = np.dot(P[action, i, :], R[action, i, :])

	for t in reversed(range(T - 1)):
		for i in range(env.nS):
			candidates = np.empty(n_actions)
			for a in range(n_actions):
				candidates[a] = np.dot(P[a, i, :], R[a, i, :] + V[:, t + 1])
			action = np.argmax(candidates)
			policy[i, t] = action
			V[i, t] = np.dot(P[action, i, :], R[action, i, :] + V[:, t + 1])
	return(policy, V)

policy, V = fh_policy(P, R, T)

def expected_reward(policy, init_state=0):
	n_actions = env.nA
	n_states = env.nS

	trans_probs = []
	for t in range(T - 1):
		rows = []
		for i in range(n_states):
			a = policy[i, t]
			rows.append(P[a, i, :])
		trans_probs.append(np.vstack(rows))
	
	action = policy[init_state, 0]
	exp_rew = np.dot(P[action, init_state, :], R[action, init_state, :])

	mat = np.eye(n_states)
	for t in range(T - 1):
		mat = mat @ trans_probs[t]
		vec = np.empty(n_states)
		for i in range(n_states):
			action = policy[i, t + 1]
			vec[i] = np.dot(P[action, i, :], R[action, i, :])
		exp_rew += mat[init_state, :] @ vec
	return(exp_rew)
		
# sanity check: expected reward of optimal policy should equal the first column of the V matrix
test1 = np.array([expected_reward(policy, init) for init in range(64)])
test2 = np.array([V[init, 0] for init in range(64)])
(np.isclose(test1, test2, 1e-4)).all()

# Policy evaluation: here's where YOU also code
"""
Insert here your code to evaluate
the total expected rewards over the planning horizon T
if one follows your policy. Do the same for a random policy (i.e. the
sample policy given above). As a sanity check, your policy should get an
expected reward of at least the one obtained by the random policy!
"""
# construct random policy by sampling action space
rand_policy = np.zeros((n_states, T), dtype=int)
for (i, t) in zip(range(n_states), range(T)):
	rand_policy[i, t] = env.action_space.sample()

# compute expected rewards for optimal and random policy
expected_reward(rand_policy)
expected_reward(policy)


