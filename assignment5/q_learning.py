import numpy as np
import gym
import scipy.optimize as optimize

def learn_policy(env, alpha, decay_schedule, n_runs=int(1e3), q_init=1):

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = q_init*np.ones((n_states, n_actions))

    for i in range(n_runs):
    
        state_process = list()
        action_process = list()
        state_process.append(env.reset())
        t = 0
        done = False

        while not done:

            learning_rate = decay_schedule(1 + t)

            if (np.max(Q[state_process[t], :]) - np.min(Q[state_process[t], :])) < 1e-7:
                action_process.append(np.random.choice(range(n_actions)))
            else:
                action_process.append(np.argmax(Q[state_process[t], :]))
        
            next_state, reward, done, _ = env.step(action_process[t])

            state_process.append(next_state)

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

    state = env.reset()

    for i in range(n_sim):

        print(env.render())

        state, reward, done, _ = env.step(policy[int(state)])
        
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

        state = env.reset()
        done = False
        
        while not done:
            state, reward, done, _ = env.step(policy[int(state)])
            if state == 63:
                n_successes += 1
        
    return(n_successes/n_runs)

class KSchedule:

    def __init__(self, omega, K):
        assert omega <= 1 and omega > 0.5
        self.omega = omega
        self.K = K

        self.min_t = 0
        done = False

        while not done:
            self.min_t += 1
            done = self.K/(self.min_t + self.K)**self.omega < 1


    def __call__(self, t):
        return(self.K/(self.min_t + t - 1 + self.K)**self.omega)

def poly_schedule(t, omega):
    assert omega <= 1 and omega > 0.5
    return(1/t**omega)

