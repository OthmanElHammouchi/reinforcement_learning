import gym
import numpy as np

env = gym.make("FrozenLake8x8-v1", desc= None, is_slippery = True).env

def avg_pi(P, R, init_policy):

    n_actions, n_states, _ = P.shape

    policy = init_policy.copy()

    changed = True

    while changed:

        # construct transition matrix and reward vector for current policy
        Pf = np.zeros((n_states, n_states))

        for i in range(n_states):
            Pf[i, :] = P[policy[i], i, :]

        rf = np.zeros(n_states)
        for i in range(n_states):
            rf[i] = np.dot(R[policy[i], i, :], P[policy[i], i, :])

        # solve linear system to obtain average reward and u0
        mat = np.block([
            [np.eye(n_states) - Pf, np.zeros((n_states, n_states)), np.zeros((n_states, n_states))],
            [np.eye(n_states), np.eye(n_states) - Pf, np.zeros((n_states, n_states))],
            [np.zeros((n_states, n_states)), np.eye(n_states), np.eye(n_states) - Pf]
        ])

        rhs = np.concatenate([np.zeros(n_states), rf, np.zeros(n_states)])

        res = np.linalg.lstsq(mat, rhs)

        avg_reward = res[0][:n_states]
        u0 = res[0][n_states:2*n_states]
        
        tol = 1e-5

        # construct set of candidate actions for every state; if non-empty,
        # update policy by selecting one of these at random and set "changed" to true
        changed = False
        for i in range(n_states):
            candidates = []
            for a in range(n_actions):

                ria = np.dot(R[a, i, :], P[a, i, :])

                condition = np.dot(avg_reward, P[a, i, :]) - avg_reward[i] > tol or (np.abs(np.dot(avg_reward, P[a, i, :]) - avg_reward[i]) < tol and (ria + np.dot(u0, P[a, i, :]) - avg_reward[i] - u0[i] > tol))

                if condition:
                    candidates.append(a)
                    
            if len(candidates) > 0:
                policy[i] = np.random.choice(candidates)
                changed = True
    
    # compute average reward corresponding to optimal policy
    Pf = np.zeros((n_states, n_states))

    for i in range(n_states):
        Pf[i, :] = P[policy[i], i, :]

    rf = np.zeros(n_states)
    for i in range(n_states):
        rf[i] = np.dot(R[policy[i], i, :], P[policy[i], i, :])

    mat = np.block([
            [np.eye(n_states) - Pf, np.zeros((n_states, n_states)), np.zeros((n_states, n_states))],
            [np.eye(n_states), np.eye(n_states) - Pf, np.zeros((n_states, n_states))],
            [np.zeros((n_states, n_states)), np.eye(n_states), np.eye(n_states) - Pf]
        ])

    rhs = np.concatenate([np.zeros(n_states), rf, np.zeros(n_states)])

    res = np.linalg.lstsq(mat, rhs)

    avg_reward = res[0][:n_states]
    u0 = res[0][n_states:2*n_states]

    return(policy, avg_reward, u0)

n_actions = env.nA
n_states = env.nS

P = np.zeros((n_actions, n_states, n_states))
R = np.zeros((n_actions, n_states, n_states))

for a in range(n_actions):
	for i in range(n_states):
		for (p, j, r, d) in env.P[i][a]:
			P[a, i, j] += p
			R[a, i, j] += r

init_policy = np.zeros(n_states, dtype=int)
policy, avg_reward, u0 = avg_pi(P, R, init_policy)



