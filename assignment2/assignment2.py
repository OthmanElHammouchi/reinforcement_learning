import gym
import numpy as np

# We will load a DiscreteEnv and retrieve the probability and reward
# information
env = gym.make("FrozenLake8x8-v1", desc=None, map_name=None).env
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
alpha = 0.999
n_actions = env.nA
n_states = env.nS

P = np.zeros((n_actions, n_states, n_states))
R = np.zeros((n_actions, n_states, n_states))

for a in range(n_actions):
	for i in range(n_states):
		for (p, j, r, d) in env.P[i][a]:
			P[a, i, j] += p
			R[a, i, j] += r

def pi(P, R, alpha = 0.5, maxiter=1e3):
    #generate arbitrary initial policy
    policy = np.zeros(n_states, dtype=int)
    for i in range(n_states):
        policy[i] = env.action_space.sample()
    
    iter = 0
    while iter < maxiter:
        #solve for the value function
        transition_matrix = np.zeros((n_states, n_states))
        for i in range(n_states):
            transition_matrix[i, :] = P[policy[i], i, :]

        rhs_vector = np.zeros(n_states)
        for i in range(n_states):
            rhs_vector[i] = np.dot(transition_matrix[i, :], R[policy[i], i, :])

        value = np.linalg.inv(np.eye(n_states) - alpha*transition_matrix) @ rhs_vector

        #improve the policy
        changes = np.zeros((n_states, n_actions))
        for action in range(n_actions):
            reward_vector = np.zeros(n_states)
            for i in range(n_states):
                reward_vector[i] = np.dot(P[action, i, :], R[action, i, :])
            changes[:, action] = reward_vector + alpha*(P[action, :, :] @ value) - value

        if np.all(changes <= 0):
            return(policy)

        policy = np.argmax(changes, axis=1)
        iter += 1
    return(policy)

def pi_improved(P, R, alpha = 0.5, maxiter=1e3):
    # first iteration

    ## generate arbitrary initial policy
    policy = np.zeros(n_states, dtype=int)
    for i in range(n_states):
        policy[i] = env.action_space.sample()
    
    ## evaluate policy
    Pf = np.zeros((n_states, n_states))
    for i in range(n_states):
        Pf[i, :] = P[policy[i], i, :]

    rf = np.zeros(n_states)
    for i in range(n_states):
        rf[i] = np.dot(Pf[i, :], R[policy[i], i, :])

    inverse = np.linalg.inv(np.eye(n_states) - alpha*Pf)
    value =  inverse @ rf

    ## improve policy
    changes = np.zeros((n_states, n_actions))
    for action in range(n_actions):
        reward = np.zeros(n_states)
        for i in range(n_states):
            reward[i] = np.dot(P[action, i, :], R[action, i, :])
        changes[:, action] = reward + alpha*(P[action, :, :] @ value) - value

    if np.all(changes <= 0):
        return(policy)

    new_policy = np.argmax(changes, axis=1)

    ## get change indices for Sherman-Woodbury
    change_indices = np.asarray(new_policy != policy).nonzero()[0]
    policy = new_policy

    # main loop
    n = 1
    while n < maxiter:
        ## evaluate policy using Sherman-Woodbury
        Pg = np.zeros((n_states, n_states))
        for i in range(n_states):
            Pg[i, :] = P[policy[i], i, :]

        rg = np.zeros(n_states)
        for i in range(n_states):
            rg[i] = np.dot(P[policy[i], i, :], R[policy[i], i, :])

        n_changes = len(change_indices)
        V = np.zeros((n_changes, n_states))
        for i in range(n_changes):
            V[i, :] = -alpha*(Pg[change_indices[i], :] - Pf[change_indices[i], :])

        U = np.zeros((n_states, n_changes))
        for i in range(n_changes):
            U[change_indices[i], i] = 1

        inverse = inverse - np.linalg.multi_dot([inverse, U, np.linalg.inv(np.eye(n_changes) + np.linalg.multi_dot([V, inverse, U])), V, inverse])

        value = inverse @ rg

        ## improve policy
        changes = np.zeros((n_states, n_actions))
        for action in range(n_actions):
            reward_vector = np.zeros(n_states)
            for i in range(n_states):
                reward_vector[i] = np.dot(P[action, i, :], R[action, i, :])
            changes[:, action] = reward_vector + alpha*(P[action, :, :] @ value) - value

        if np.all(changes <= 0):
            return(policy)

        new_policy = np.argmax(changes, axis=1)

        ## get change indices for Sherman-Woodbury
        change_indices = np.asarray(new_policy != policy).nonzero()[0]
        policy = new_policy
        Pf = Pg

        n += 1
    return(policy)

from timeit import default_timer as timer

t1 = timer()
policy1 = pi(P, R, alpha)
t2 = timer()

pi_time = t2 - t1

t1 = timer()
policy2 = pi_improved(P, R, alpha)
t2 = timer()

pi_improved_time = t2 - t1

print('Basic policy iteration time: ', pi_time, '\nImproved policy iteration time: ', pi_improved_time)

n_sim = int(1e4)

n_successes = 0
for n in range(n_sim): 
    state = env.reset()
    while True:
        action = policy[state]
        state, reward, done, _ = env.step(action)
        if done:
            if state == 63:
                n_successes +=1
                break
            else:
                break
    env.close()

print('Success rate: ', n_successes/n_sim)