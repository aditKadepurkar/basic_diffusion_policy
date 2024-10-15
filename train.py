"""
This file trains the diffusion policy using the demonstrations
collected from the human demonstrator.
"""

import argparse
import os
import json
import equinox as eqx
from diffusion.diffusion_policy import DiffusionPolicy
import jax.numpy as jnp
import jax
from PIL import Image
import numpy as np
from diffusion.data_loader import DataLoader

def train_diffusion_policy(demonstrations_path, output_dir, config_path):

    a = jnp.array([[1.4, 1.1, 2.3, 3.8, 5.2],
                   [0.4, 0.8, 2.8, 4.1, 4.8],
                   [-1.2, 1.4, 2.5, 4.3, 5.0],
                   [3.1, 2.5, 3.4, 4.3, 5.3],
                   [1.2, 3.4, 5.4, 4.8, 5.1]])

    # image = Image.open('test_image.jpg')

    # a = jnp.array(image) / 255.0
    
    # print(a)

    

    key = jax.random.PRNGKey(0)

    data_path = "data/1728922451_627212/demo.hdf5"

    policy = DiffusionPolicy(key=key, data_path=data_path)


    # a_out = policy.forward_diffusion(a, 1000)
    
    # print(a_out)






    # now I test the prediction function for errors
    # obs = jnp.array([0.8, 1.4, 1.3, 4.2, 2.3, 3.5, 0.3]) # 7 dimensional observation
    # val = policy.predict_action(obs, 1000, 2, 7)
    # print(val)



    print("Training the policy")

    policy.train()




train_diffusion_policy("demonstrations.hdf5", "output_dir", "config.json")