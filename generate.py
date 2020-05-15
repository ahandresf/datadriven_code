import numpy as np
import pandas as pd
import matplotlib as plt
import random
import sys
sys.path.append("../../")
from simulator_integration import Model
from star import Star
import policies
from sklearn.model_selection import train_test_split
import os
import pickle
import time

def generate_data(total_sample=100000, config=None):
    action_space_dim = 1
    state_space_dim = 4

    action_min = -3
    action_max = 3

    model = Model()
    star = Star()

    x_set = np.empty(shape=(total_sample, int(action_space_dim+state_space_dim)))
    y_set = np.empty(shape=(total_sample, int(state_space_dim)))

    n = 0
    while True:
        # episode start
        config = star.brain_config_to_sim_config(test_config)
        model.simulator_reset(config)
        sim_state = model.get_sim_state()
        brain_state = star._get_brain_state(sim_state)
        if n > total_sample-1:
            break
        else:
            pass 
        for t in range(config["max_iterations"]):
            if n > total_sample-1:
                break
                print('maximum number of samples recoded from the environment')
            else:
                pass 
            action = policies.random_policy(brain_state)
            sim_action = star.brain_action_to_sim_action(action)
            data_set = np.append(np.array([brain_state["theta"], brain_state["alpha"], brain_state["theta_dot"], brain_state["alpha_dot"]]), np.array([action["Vm"]]))
            model.simulator_step(action)
            sim_state = model.get_sim_state()
            
            x_set[n, :] = data_set
            
            model.simulator_step(action)
            sim_state = model.get_sim_state()
            brain_state, reward, terminal, terminal_type = star.compute_srt(sim_state, action, sim_action, t)
            #model.render()
            
            y_set[n, :] = np.array([brain_state["theta"], brain_state["alpha"], brain_state["theta_dot"], brain_state["alpha_dot"]])
            
            n += 1
            if terminal:
                print("Episode finished after {} timesteps".format(t+1))
                break
    os.chdir('./datadrivenmodel')
    with open('./env_data/x_set.pickle', 'wb') as f:
        pickle.dump(x_set, f, pickle.HIGHEST_PROTOCOL)
    with open('./env_data/y_set.pickle', 'wb') as f:
        pickle.dump(y_set, f, pickle.HIGHEST_PROTOCOL)
    return x_set, y_set

if __name__ == '__main__':
    test_config = {
        # Parameters
        "Lp": 0.129,
        "mp": 0.024,
        
        # Initial Conditions
        "theta": random.randint(-80, 80) * 2 * np.pi / 360,
        "alpha": 0 + np.random.randn() * 0.01, # make sure pi if reset_down
        "theta_dot": 0 + np.random.randn() * 0.01,
        "alpha_dot": 0 + np.random.randn() * 0.01,
        
        "max_iterations": 2048
        }
    os.chdir('../../')
    x_set, y_set = generate_data(total_sample=100000, config=test_config)
    #x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, test_size=0.33, random_state=42)