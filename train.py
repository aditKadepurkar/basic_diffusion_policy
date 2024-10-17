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
import optax

def train(Policy, lr=1e-3, epochs=100):
    
    model = Policy.model
    data_loader = Policy.data_loader
    optim = optax.adamw(learning_rate=lr)

        # print(model)
    for e in range(epochs):
        params = eqx.filter(model, eqx.is_inexact_array)
        # print(f"Initial Params: {params}")
        
        opt_state = optim.init(params)

        # Load data
        for data in data_loader.load_data_in_batches():
            observations = data['states']
            expert_action_sequence = data['actions']

            # print(observations[0].shape, expert_action_sequence.shape)

            # Loop over each observation
            loss_period = 0
            accumulated_grads = None
            for i in range(len(observations)):
                a_t, loss_value, grads = DiffusionPolicy.predict_action(
                    model=model,
                    observation=jnp.copy(observations[i]),
                    Y=jnp.array(expert_action_sequence[i]),
                    key=jax.random.PRNGKey(0),
                    T=1000,
                    n_actions=4
                )
                # print(grads)
                loss_period += loss_value
                
                if accumulated_grads is None:
                    accumulated_grads = grads
                else:
                    # Accumulate gradients across steps
                    accumulated_grads = jax.tree_util.tree_map(lambda g1, g2: g1 + g2, accumulated_grads, grads)

                
                # if (i + 1) % 4 == 0:
                    # loss_period /= 4
                    # print(f"Loss: {loss_period}")
                # print(f"Gradients: {grads}")
            
            loss_period /= len(observations)        
            
            params = eqx.filter(model, eqx.is_array)
            updates, opt_state = optim.update(accumulated_grads, opt_state, params)
            model = eqx.apply_updates(model, updates)
        print(f"Epoch: {e}, Loss: {loss_period}")

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

    train(Policy=policy, lr=0.001)




train_diffusion_policy("demonstrations.hdf5", "output_dir", "config.json")