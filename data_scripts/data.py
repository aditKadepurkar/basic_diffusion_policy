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



DATA_FILE = "demonstrations/demo_norm.hdf5"
OUT_FILE = "demonstrations/demo.hdf5"

count_total = 0
count_action = 0
count_obs = 0

# hdf5 files that we are using are in the format:
# data
#   - demo_1, demo_2, ...
#       - states, actions

with h5py.File(DATA_FILE, "r") as fin:
    data = fin['data']
    with h5py.File(OUT_FILE, "w") as fout:
        fout.create_group('data')
        for demo in data:
            demo_group = fout['data'].create_group(demo)
            actions = data[demo]['actions']
            states = data[demo]['states']
            demo_group.create_dataset('actions', data=actions, chunks=True, maxshape=(None, actions.shape[1]))
            demo_group.create_dataset('states', data=states, chunks=True, maxshape=(None, states.shape[1]))
            for i in range(actions.shape[0]):
                action = actions[i]
                state = states[i]
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
