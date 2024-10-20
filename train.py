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
from eval import eval_policy

def train(Policy, lr=1e-4, epochs=100):

    model = Policy.model
    data_loader = Policy.data_loader
    optim = optax.adamw(learning_rate=lr)



        # print(model)
    for e in range(epochs):
        max_loss = 0
        params = eqx.filter(model, eqx.is_inexact_array)
        # print(f"Initial Params: {params}")
        
        opt_state = optim.init(params)

        # Load data
        for data in data_loader:
            observations = data['states']
            expert_action_sequence = data['actions']

            # print(observations.shape, expert_action_sequence.shape)

            # Loop over each observation
            
            # for i in range(len(observations)):
            a_t, loss_value, grads = DiffusionPolicy.predict_action(
                model=model,
                observation=jnp.copy(observations),
                Y=jnp.array(expert_action_sequence),
                key=jax.random.PRNGKey(0),
                T=50,
                n_actions=4,
                alpha=Policy.alpha
            )
            # print(grads)
            # loss_period += loss_value

            # if accumulated_grads is None:
                # accumulated_grads = grads
            # else:
                # Accumulate gradients across steps
                # accumulated_grads = jax.tree_util.tree_map(lambda g1, g2: g1 + g2, accumulated_grads, grads)

                
                # if (i + 1) % 4 == 0:
                    # loss_period /= 4
                    # print(f"Loss: {loss_period}")
                # print(f"Gradients: {grads}")

            # loss_value /= len(observations)
            if loss_value > max_loss:
                max_loss = loss_value

            params = eqx.filter(model, eqx.is_array)
            updates, opt_state = optim.update(grads, opt_state, params)
            model = eqx.apply_updates(model, updates)
            # print(f"Loss: {loss_period}")
        print(f"Epoch: {e+1}, Loss: {loss_value}, Max Loss: {max_loss}")
    print("Training complete")

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

    data_path = "demonstrations/demo.hdf5"

    policy = DiffusionPolicy(key=key, data_path=data_path)

    print("Training the policy")

    # policy.train()

    lr_schedule = optax.schedules.exponential_decay(1e-3, 100, 0.9)

    train(Policy=policy, lr=lr_schedule, epochs=100)


    print("Eval")

    eval_policy(Policy=policy)




train_diffusion_policy("demonstrations.hdf5", "output_dir", "config.json")
