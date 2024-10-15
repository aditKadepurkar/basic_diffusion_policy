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

def train_diffusion_policy(demonstrations_path, output_dir, config_path):

    a = jnp.array([[1.4, 1.1, 2.3, 3.8, 5.2],
                   [0.4, 0.8, 2.8, 4.1, 4.8],
                   [-1.2, 1.4, 2.5, 4.3, 5.0],
                   [3.1, 2.5, 3.4, 4.3, 5.3],
                   [1.2, 3.4, 5.4, 4.8, 5.1]])

    # image = Image.open('test_image.jpg')

    # a = jnp.array(image) / 255.0
    
    print(a)
    # print()

    key = jax.random.PRNGKey(0)

    policy = DiffusionPolicy(key=key)


    a_out = policy.forward_diffusion(a, 1000)

    # a_out.clip(0.0, 1.0)

    # a_out = (a_out * 255.0).astype(np.uint8)
    # a_out = np.array(a_out)

    # image_out = Image.fromarray(a_out)
    # image_out.save('test_image_out.jpg')
    
    print(a_out)




    # now I test the prediction function for errors
    obs = jnp.array([0.8, 1.4, 1.3, 4.2, 2.3, 3.5, 0.3]) # 7 dimensional observation
    val = policy.predict_action(obs, 1000, 1, 7)
    print(val)

train_diffusion_policy("demonstrations.hdf5", "output_dir", "config.json")