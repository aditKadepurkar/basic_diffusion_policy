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



DATA_FILE = "demonstrations/1731007425_4282627/demo.hdf5"
OUT_FILE = "demonstrations/data_norm.hdf5"

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
            states = jnp.array(data[demo]['states'])
            normalized_actions = normalize_data(actions)
            normalized_states = normalize_data(states)
            
            demo_group.create_dataset('actions', data=normalized_actions, chunks=True, maxshape=(None, normalized_actions.shape[1]))
            demo_group.create_dataset('states', data=normalized_states, chunks=True, maxshape=(None, normalized_states.shape[1]))



with h5py.File(OUT_FILE, "r") as fout, h5py.File(DATA_FILE, "r") as fin:
    print(fin.keys())
    print(fin['data'].keys(), fout['data'].keys())
    print(fin['data']['demo_1']['actions'].shape, fout['data']['demo_1']['actions'].shape)
    print(fin['data']['demo_1']['states'].shape, fout['data']['demo_1']['states'].shape)

