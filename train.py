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


def train(Policy):
    model = Policy.model
    data_loader = Policy.data_loader
    optim = Policy.optim

    params = eqx.filter(model, eqx.is_inexact_array)
    # print(f"Initial Params: {params}")
    opt_state = optim.init(params)

    # Load data
    data = data_loader.load_data(count=1)
    for demo in data:
        observations = data[demo]['states']
        expert_action_sequence = data[demo]['actions']

        # Loop over each observation
        loss_period = 0
        for i in range(len(observations)):
            a_t, loss_value, grads = DiffusionPolicy.predict_action(
                model=model,
                observation=jnp.copy(observations[i]),
                Y=jnp.array(expert_action_sequence[i]),
                key=jax.random.PRNGKey(0),
                T=1000,
                n_actions=4
            )
            loss_period += loss_value
            if (i + 1) % 4 == 0:
                loss_period /= 4
                print(f"Loss: {loss_period}")
            # print(f"Gradients: {grads}")

                params = eqx.filter(model, eqx.is_array)
                updates, opt_state = optim.update(grads, opt_state, params)
                model = eqx.apply_updates(model, updates)

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

    print("Training the policy")

    # policy.train()

    train(policy)




train_diffusion_policy("demonstrations.hdf5", "output_dir", "config.json")