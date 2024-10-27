"""
Reads in hdf5 file and normalizes the data
"""


# open file and normalize data
"""
In this file we analyze the data and clean it

Currently has a bug 10/21/2024
"""


import sys
import os

sys.path.append(os.getcwd())

import h5py
from jax import numpy as jnp
from diffusion.data_loader import DataLoader



DATA_FILE = "demonstrations/1729535071_8612459/demo.hdf5"
OUT_FILE = "demonstrations/demo_norm.hdf5"

count_total = 0
count_action = 0
count_obs = 0

# hdf5 files that we are using are in the format:
# data
#   - demo_1, demo_2, ...
#       - states, actions

def normalize_data(data, epsilon=1e-8):
    mean = jnp.mean(data, axis=0)
    std = jnp.std(data, axis=0)
    # print(mean.shape, std.shape)
    return (data - mean) / (std + epsilon)

with h5py.File(DATA_FILE, "r") as fin:
    data = fin['data']
    with h5py.File(OUT_FILE, "w") as fout:
        fout.create_group('data')
        for demo in data:
            demo_group = fout['data'].create_group(demo)
            actions = jnp.array(data[demo]['actions'])
            states = jnp.array(data[demo]['states'][:])
            
            # Normalize actions and states
            normalized_actions = normalize_data(actions)
            normalized_states = normalize_data(states)
            
            demo_group.create_dataset('actions', data=normalized_actions, chunks=True, maxshape=(None, normalized_actions.shape[1]))
            demo_group.create_dataset('states', data=normalized_states, chunks=True, maxshape=(None, normalized_states.shape[1]))
            
            for i in range(actions.shape[0]):
                action = normalized_actions[i]
                state = normalized_states[i]
                if not (jnp.linalg.norm(action) < 0.01):
                    demo_group['actions'].resize((demo_group['actions'].shape[0] + 1), axis=0)
                    demo_group['actions'][-1] = action
                    demo_group['states'].resize((demo_group['states'].shape[0] + 1), axis=0)
                    demo_group['states'][-1] = state
                else:
                    count_action += 1
                count_total += 1

print(f"Total number of data points: {count_total}")
print(f"Number of actions with 0: {count_action}")
print(f"Number of observations with 0: {count_obs}")

with h5py.File(OUT_FILE, "r") as fin:
    print(fin.keys())
    print(fin['data'].keys())
    print(fin['data']['demo_1']['actions'].shape)
    print(fin['data']['demo_1']['states'].shape)

