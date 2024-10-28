
import jax
import jax.numpy as jnp
import json
import equinox as eqx
from diffusion.cnn_policy_network import CnnDiffusionPolicy
from eval import eval_policy
from diffusion.diffusion_policy import NoiseScheduler

def make(*, action_dim, obs_dim):
    return CnnDiffusionPolicy(
        action_dim=action_dim,
        obs_dim=64,
    )

def load(filename):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = make(**hyperparams)
        return eqx.tree_deserialise_leaves(f, model)

model = load("model")
print(model)
print("Model loaded successfully!")

def alpha_bar_fn(t):
        return jnp.cos((t + 0.008) / 1.008 * jnp.pi / 2) ** 2

betas = []
for i in range(50):
    t1 = i / 50
    t2 = (i + 1) / 50
    betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), 0.999))
betas = jnp.array(betas)

noise_scheduler = NoiseScheduler(50, betas)

eval_policy(model=model, noise_scheduler=noise_scheduler)

