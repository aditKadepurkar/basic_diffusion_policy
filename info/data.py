"""
In this file we analyze the data
"""


import sys
import os

sys.path.append(os.getcwd())

import h5py
from jax import numpy as jnp
from diffusion.data_loader import DataLoader



DATA_FILE = "demonstrations/1729359268_7140367/demo.hdf5"

count_total = 0
count_action = 0
count_obs = 0

for data in DataLoader(DATA_FILE, "data", 32):
    # print(data['actions'].shape)
    d_0 = data['actions'][0]
    o_0 = data['states'][0]
    for i in range(1, data['actions'].shape[0]):
        d_1 = data['actions'][i]
        o_1 = data['states'][i]
        
        if (d_0 < 0.01).all():
            count_action += 1
        
        if (jnp.mean(jnp.square(o_0 - o_1)) < 0.01).all():
            count_obs += 1
        
        count_total += 1
        
        # MSE_a = jnp.mean(jnp.square(d_0 - d_1))
        # MSE_o = jnp.mean(jnp.square(o_0 - o_1))
        # print(f"MSE between data {i-1} and {i}: {MSE_a}, {MSE_o}")
        
        # d_0 = d_1
        o_0 = o_1
        
print(f"Total number of data points: {count_total}")
print(f"Number of actions with 0: {count_action}")
print(f"Number of observations with 0: {count_obs}")