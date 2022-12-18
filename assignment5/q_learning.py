import numpy as np
import gym
import scipy.optimize as optimize

def learn_policy(env, alpha, n_runs=int(1e3), q_init=1, omega=0.75, epsilon=0, K=1):

    assert omega < 1 and omega > 0.5

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = q_init*np.ones((n_states, n_actions))

    for t in range(n_runs):
    
        state_process = np.zeros(n_runs + 1, dtype=int)
        action_process = np.zeros(n_runs, dtype=int)

        state_process[0], _ = env.reset()
        t = 0
        done = False

        min_t = 0
        small_enough = False

        while not small_enough:
            min_t+=1
            small_enough = K/(min_t+K) < 1

        while not done:
            
            learning_rate = K/(min_t + t + K)#**omega

            if np.random.uniform() <= 1 - epsilon:
                action_process[t] = np.argmax(Q[state_process[t], :])
            else:
                action_process[t] = np.random.choice(range(n_actions))
        
            state_process[t + 1], reward, done, _ , _ = env.step(action_process[t])

            Q[state_process[t], action_process[t]] = (1 - learning_rate)*Q[state_process[t], action_process[t]] + learning_rate * (reward + alpha*np.max(Q[state_process[t + 1], :]))

            t += 1

    policy = np.array([np.argmax(Q[state, :]) for state in range(n_states)])

    return(Q, policy)



def gen_rand_policy(env):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    rand_policy = np.array([np.random.choice(range(n_actions)) for state in range(n_states)])

    return(rand_policy)

def sim_policy(env, policy, n_sim=int(1e3)):

    state, _ = env.reset()

    for i in range(n_sim):

        print(env.render())

        state, reward, done, _, _ = env.step(policy[int(state)])
        
        if done:
            
            if state == 63:
                print("Succes")
            else:
                print("Failure")
            
            env.close()
            break
    return

def eval_policy(env, policy, n_runs=int(1e3)):

    n_successes = 0

    for i in range(n_runs):

        state, _ = env.reset()
        done = False
        
        while not done:
            state, reward, done, _, _ = env.step(policy[int(state)])
            if state == 63:
                n_successes += 1
        
    return(n_successes/n_runs)
