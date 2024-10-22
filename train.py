"""
This file trains the diffusion policy using the demonstrations
collected from the human demonstrator.
"""

import argparse
import os
import json
import equinox as eqx
from diffusion.diffusion_policy import DiffusionPolicy, NoisePredictionNetwork, NoiseScheduler
import jax.numpy as jnp
import jax
from PIL import Image
import numpy as np
from diffusion.data_loader import DataLoader
import optax
from eval import eval_policy
from diffusion.mlp_model import MLP
from functools import partial
import tqdm
from tqdm import trange
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion.cnn_policy_network import CnnDiffusionPolicy





def train(noise_pred_nw, noise_scheduler, dataloader, epochs):

    lr = optax.cosine_decay_schedule(1e-4, 100)
    # alpha = 
    # gamma = 

    optim = optax.adamw(learning_rate=lr)

    key = jax.random.key(0)


    for e in trange(epochs, desc="Epochs"):
        # max_loss_value = [0]
        params = eqx.filter(noise_pred_nw, eqx.is_inexact_array)
        opt_state = optim.init(params)

        # Iterate over batches
        for i, data in zip(trange(dataloader.get_batch_count(), desc="Batches"), dataloader):
        
            obs = data['states']
            actions = data['actions']

            # film layer stuff



            key, subkey, subkey_2 = jax.random.split(key, 3)


            # random noise
            noise = jax.random.normal(subkey, (actions.shape))

            timesteps = jax.random.randint(subkey_2, (actions.shape[0],), 0, 50)


            # forward diffusion
            noise_actions = jax.vmap(noise_scheduler.add_noise)(actions, noise, timesteps)

            # @partial(jax.jit, static_argnames=['noise_pred_nw'])
            def loss(noise_pred_nw, actions, T, observations):
                """
                observations: (batch, observation_dim)
                actions: (batch, n_actions)
                
                """
                
                # print(actions.shape, T.shape, observations.shape)
                # print(observations)

                batch, n_actions = actions.shape
                # k = jnp.broadcast_to(T[:, None], (batch, n_actions))
                # k = jnp.reshape(T, (batch, 1))

                # X = jnp.concatenate([actions, observations, k], axis=1)



                noise_pred = jax.vmap(noise_pred_nw)(actions, T, observations)

                loss = optax.l2_loss(noise_pred, noise)

                return loss

                # ret = jnp.square(noise - noise_pred)


                # ret_max = jnp.max(ret)
                # print(ret_max)
                # return jnp.mean(ret)
            
            loss_value, grads = eqx.filter_value_and_grad(loss)(
                noise_pred_nw, 
                noise_actions, 
                timesteps, 
                obs,)
            
            # i.set_postfix(loss=loss_value)

            # print(loss_value)
            # print(grads)

            params = eqx.filter(noise_pred_nw, eqx.is_inexact_array)
            updates, opt_state = optim.update(grads, opt_state, params)
            noise_pred_nw = eqx.apply_updates(noise_pred_nw, updates)
        dataloader.shuffle_data()



        print(f"Epoch: {e+1}, Loss: {loss_value}")











# def train(Policy, lr=1e-4, epochs=100):

#     model = Policy.model
#     data_loader = Policy.data_loader
#     optim = optax.adamw(learning_rate=lr)



#         # print(model)
#     for e in range(epochs):
#         max_loss = 0
#         params = eqx.filter(model, eqx.is_inexact_array)
#         # print(f"Initial Params: {params}")
        
#         opt_state = optim.init(params)

#         # Load data
#         for data in data_loader:
#             observations = data['states']
#             expert_action_sequence = data['actions']

#             # print(observations.shape, expert_action_sequence.shape)

#             # Loop over each observation
            
#             # for i in range(len(observations)):
#             a_t, loss_value, grads = DiffusionPolicy.predict_action(
#                 model=model,
#                 observation=jnp.copy(observations),
#                 Y=jnp.array(expert_action_sequence),
#                 key=jax.random.PRNGKey(0),
#                 T=50,
#                 n_actions=4,
#                 alpha=Policy.alpha
#             )
#             # print(grads)
#             # loss_period += loss_value

#             # if accumulated_grads is None:
#                 # accumulated_grads = grads
#             # else:
#                 # Accumulate gradients across steps
#                 # accumulated_grads = jax.tree_util.tree_map(lambda g1, g2: g1 + g2, accumulated_grads, grads)

                
#                 # if (i + 1) % 4 == 0:
#                     # loss_period /= 4
#                     # print(f"Loss: {loss_period}")
#                 # print(f"Gradients: {grads}")

#             # loss_value /= len(observations)
#             if loss_value > max_loss:
#                 max_loss = loss_value

#             params = eqx.filter(model, eqx.is_array)
#             updates, opt_state = optim.update(grads, opt_state, params)
#             model = eqx.apply_updates(model, updates)
#             # print(f"Loss: {loss_period}")
#         print(f"Epoch: {e+1}, Loss: {loss_value}, Max Loss: {max_loss}")
#     print("Training complete")

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

    data_path = "demonstrations/1729535071_8612459/demo.hdf5"

    policy = DiffusionPolicy(key=key, data_path=data_path)

    print("Training the policy")

    # policy.train()

    # lr_schedule = optax.schedules.exponential_decay(1e-3, 100, 0.9)

    # train(Policy=policy, lr=lr_schedule, epochs=100)

    # noise_pred_nw = NoisePredictionNetwork(7, 40)
    # noise_pred_nw = MLP(157, 7*4)
    noise_pred_nw = CnnDiffusionPolicy(action_dim=7, obs_dim=30*4)


    def alpha_bar_fn(t):
        return jnp.cos((t + 0.008) / 1.008 * jnp.pi / 2) ** 2

    betas = []
    for i in range(50):
        t1 = i / 50
        t2 = (i + 1) / 50
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), 0.999))
    betas = jnp.array(betas)

    noise_scheduler = NoiseScheduler(50, betas)


    train(noise_pred_nw=noise_pred_nw, 
          noise_scheduler=noise_scheduler, 
          dataloader=DataLoader(data_path, "data", 32), 
          epochs=100)

    print("Eval")

    eval_policy(model=noise_pred_nw)




train_diffusion_policy("demonstrations.hdf5", "output_dir", "config.json")
