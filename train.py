"""
This file trains the diffusion policy using the demonstrations
collected from the human demonstrator.
"""

import argparse
import os
import json
import equinox as eqx
from diffusion.diffusion_policy import DiffusionPolicy, NoiseScheduler
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
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion.cnn_policy_network import CnnDiffusionPolicy





def train(noise_pred_nw, noise_scheduler, dataloader, epochs):
    # print(dataloader.get_batch_count())
    lr = optax.warmup_cosine_decay_schedule(
        init_value=1e-3,
        peak_value=1e-3,
        warmup_steps=500,
        decay_steps=dataloader.get_batch_count() * epochs,
    )

    # lr = optax.warmup_exponential_decay_schedule(
    #     init_value=1e-3,
    #     peak_value=1e-3,
    #     warmup_steps=500,
    #     decay_rate=0.05,
    #     transition_steps=dataloader.get_batch_count() * epochs,
    # )

    # lr = 1e-3
    # alpha = 
    # gamma = 

    optim = optax.adamw(learning_rate=lr)

    key = jax.random.key(0)
    params = eqx.filter(noise_pred_nw, eqx.is_inexact_array)
    opt_state = optim.init(params)
    step = 0

    with tqdm.tqdm(total=epochs, desc="Epoch") as poch:
        for e in range(epochs):
            epoch_loss = 0.0
            # Iterate over batches
            with tqdm.tqdm(dataloader, desc="Batches", total=dataloader.get_batch_count(), leave=False) as data_iter:
                for data in data_iter:
            # for i, data in zip(trange(dataloader.get_batch_count(), desc="Batches"), dataloader):
            
                    obs = data['states']
                    actions = data['actions']

                    # film layer stuff



                    key, subkey, subkey_2 = jax.random.split(key, 3)


                    # random noise
                    noise = jax.random.normal(subkey, (actions.shape)) # (batch, n_actions, action_dim)

                    timesteps = jax.random.randint(subkey_2, (actions.shape[0],), 0, 50) # (batch,)

                    del subkey, subkey_2


                    # forward diffusion
                    noise_actions = noise_scheduler.add_noise(actions, noise, timesteps) # (batch, n_actions, action_dim)

                    def loss(noise_pred_nw, actions, T, observations, noise):
                        """
                        observations: (batch, observation_dim * n_observations)
                        actions: (batch, n_actions, action_dim)
                        """
                        noise_pred = noise_pred_nw(actions, T, observations)
                        loss = jnp.mean(jnp.square(noise_pred - noise))
                        return loss

                        # ret = jnp.square(noise - noise_pred)


                        # ret_max = jnp.max(ret)
                        # print(ret_max)
                        # return jnp.mean(ret)

                    
                    loss_value, grads = eqx.filter_value_and_grad(loss)(
                        noise_pred_nw, 
                        noise_actions, 
                        timesteps, 
                        obs,
                        noise)

                    params = eqx.filter(noise_pred_nw, eqx.is_inexact_array)
                    updates, opt_state = optim.update(grads, opt_state, params)
                    noise_pred_nw = eqx.apply_updates(noise_pred_nw, updates)
                    epoch_loss += loss_value
                    data_iter.set_postfix(loss=loss_value)
                    step += 1
            poch.update(1)
            poch.set_postfix(loss=epoch_loss/dataloader.get_batch_count())
            dataloader.shuffle_data()
            # print(f"Epoch: {e+1}, Loss: {epoch_loss/dataloader.get_batch_count()}")
            # print(f"Learning Rate: {lr(step)}")

            if epoch_loss/dataloader.get_batch_count() < 1e-5:
                print("Loss is less than 1e-5. Stopping training...")
                break



        # print(f"Epoch: {e+1}, Loss: {loss_value}")


def make(*, action_dim, obs_dim):
    return CnnDiffusionPolicy(
        action_dim=action_dim,
        obs_dim=obs_dim,
    )


def save(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load(filename):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = make(**hyperparams)
        return eqx.tree_deserialise_leaves(f, model)



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

    # data_path = "demonstrations/1731007425_4282627/demo.hdf5"
    data_path = "demonstrations/data_norm.hdf5" # this is the normalized version of the above data

    # policy = DiffusionPolicy(key=key, data_path=data_path)

    print("Training the policy")

    noise_pred_nw = CnnDiffusionPolicy(action_dim=4, obs_dim=64, key=key)

    params = eqx.filter(noise_pred_nw, eqx.is_inexact_array)
    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"Number of Parameters: {param_count:e}")


    def alpha_bar_fn(t):
        return jnp.cos((t + 0.008) / 1.008 * jnp.pi / 2) ** 2

    betas = []
    for i in range(50):
        t1 = i / 50
        t2 = (i + 1) / 50
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), 0.999))
    betas = jnp.array(betas, dtype=jnp.float16)

    noise_scheduler = NoiseScheduler(50, betas)
    # noise_scheduler = DDPMScheduler(
    #     num_train_timesteps=50,
    #     beta_schedule='squaredcos_cap_v2',
    #     clip_sample=True,
    #     prediction_type='epsilon'
    # )


    train(noise_pred_nw=noise_pred_nw, 
          noise_scheduler=noise_scheduler, 
          dataloader=DataLoader(data_path, "data", 128), 
          epochs=500)
    
    print("Saving model...")
    save("model", {"action_dim": 4, "obs_dim": 128}, noise_pred_nw)
    print("Model saved!")
    
    input("Press Enter to continue to Final Evaluation...")

    print("Eval")

    eval_policy(model=noise_pred_nw, noise_scheduler=noise_scheduler)


device = jax.devices("gpu")[1]
jax._src.config.update("jax_default_device", device)


train_diffusion_policy("demonstrations.hdf5", "output_dir", "config.json")
