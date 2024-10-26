
import jax
import jax.numpy as jnp
import json
import equinox as eqx
from diffusion.cnn_policy_network import CnnDiffusionPolicy
from eval import eval_policy

def make(*, action_dim, obs_dim):
    return CnnDiffusionPolicy(
        action_dim=action_dim,
        obs_dim=obs_dim,
    )

def load(filename):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = make(**hyperparams)
        return eqx.tree_deserialise_leaves(f, model)

model = load("model")
print(model)
print("Model loaded successfully!")

eval_policy(model=model)

