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
import numpy as np
from diffusion.data_loader import DataLoader



DATA_FILE = "demonstrations/1731007425_4282627/demo.hdf5"
OUT_FILE = "demonstrations/data_norm.hdf5"

# hdf5 files that we are using are in the format:
# data
#   - demo_1, demo_2, ...
#       - states, actions

# def normalize_data(data, epsilon=1e-8):
#     mean = np.mean(data, axis=0)
#     std = np.std(data, axis=0)
#     # print(mean.shape, std.shape)
#     return (data - mean) / (std + epsilon)

def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


action_data = []
state_data = []

with h5py.File(DATA_FILE, "r") as fin:
    data = fin['data']
    
    for demo in data:
        actions = np.array(data[demo]['actions'])
        action_data.append(actions)
        
        states = np.array(data[demo]['states'])
        state_data.append(states)

action_data = np.concatenate(action_data, axis=0)
state_data = np.concatenate(state_data, axis=0)

info_action = get_data_stats(action_data)
info_state = get_data_stats(state_data)

print("Data stats:")
print("Action stats:", info_action)
print("State stats:", info_state)

with h5py.File(DATA_FILE, "r") as fin:
    data = fin['data']
    with h5py.File(OUT_FILE, "w") as fout:
        fout.create_group('data')
        for demo in data:
            demo_group = fout['data'].create_group(demo)
            actions = np.array(data[demo]['actions'])
            states = np.array(data[demo]['states'])
            normalized_actions = normalize_data(actions, info_action)
            normalized_states = normalize_data(states, info_state)
            
            demo_group.create_dataset('actions', data=normalized_actions, chunks=True, maxshape=(None, normalized_actions.shape[1]))
            demo_group.create_dataset('states', data=normalized_states, chunks=True, maxshape=(None, normalized_states.shape[1]))



with h5py.File(OUT_FILE, "r") as fout, h5py.File(DATA_FILE, "r") as fin:
    print(fin.keys())
    print(fin['data'].keys(), fout['data'].keys())
    print(fin['data']['demo_1']['actions'].shape, fout['data']['demo_1']['actions'].shape)
    print(fin['data']['demo_1']['states'].shape, fout['data']['demo_1']['states'].shape)

