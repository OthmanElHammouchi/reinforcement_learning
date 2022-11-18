import gym
import numpy as np

env = gym.make("FrozenLake8x8-v1", desc= None, map_name=None, is_slippery = False).env

alpha = 0.999
n_actions = env.nA
n_states = env.nS
eps = 0.1

P = np.zeros((n_actions, n_states, n_states))
R = np.zeros((n_actions, n_states, n_states))

for a in range(n_actions):
	for i in range(n_states):
		for (p, j, r, d) in env.P[i][a]:
			P[a, i, j] += p
			R[a, i, j] += r

def vi(P, R, alpha, eps = 1e-3, v0 = None):
    
    #get state space and action set sizes
    n_states = P.shape[1]
    n_actions = P.shape[0]

    #initialise value vector
    if v0 is None:
        v0 = np.zeros(n_states)
    v = v0
    
    #set initial values for difference and tolerance
    diff = np.inf
    tol = (1 - alpha)*eps/(2*alpha)

    # main loop
    while(diff > tol):
        v_new = np.zeros(n_states)
        for state in range(n_states):
            #compute RHS of Bellman equation with current policy for all action and take max to get next iterate
            v_new[state] = np.max(np.sum(P[:, state, :]*(R[:, state, :] + alpha*v), axis = 1), axis=0)
        #update current difference       
        diff = np.max(np.abs(v_new - v))
        v = v_new
    
    #compute policy corresponding to value vector
    policy = np.zeros(n_states)
    for state in range(n_states):
        policy[state] = np.argmax(np.sum(P[:, state, :]*(R[:, state, :] + alpha*v), axis=1), axis=0)
    return v, policy

v, policy = vi(P, R, alpha, eps)

n_sim = int(1e4)
n_successes = 0
for n in range(n_sim): 
    state = env.reset()
    while True:
        env.render()
        action = int(policy[state])
        state, reward, done, _ = env.step(action)
        if done:
            if state == 63:
                n_successes +=1
                break
            else:
                break
    env.close()

print('Success rate: ', n_successes/n_sim)


state = env.reset()
while True:
    env.render()
    action = int(policy[state])
    state, reward, done, _ = env.step(action)
    if done:
        break

import mdptoolbox

res = mdptoolbox.mdp.ValueIteration(P, R, alpha, eps)
res.run()
(res.policy == policy).all()
np.linalg.norm(res.V - v)