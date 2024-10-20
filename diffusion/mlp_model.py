"""
Basic MLP model for diffusion prediction.
Implemented in JAX and equinox.
"""


import equinox as eqx
import jax
import jax.numpy as jnp
import optax


class MLP(eqx.Module):
    layers: list
    def __init__(self, in_features, out_features=7):
        key = jax.random.PRNGKey(0)
        key0, key1, key2, key3 = jax.random.split(key, 4)



        self.layers = [
            eqx.nn.Linear(in_features, 64, key=key1, use_bias=False),
            jax.nn.relu,
            eqx.nn.Linear(64, 64, key=key2, use_bias=False),
            jax.nn.relu,
            eqx.nn.Linear(64, out_features, key=key3, use_bias=False),
        ]


    def __call__(self, x):
        # print(x.shape)
        x = x.T
        for layer in self.layers:
            # print(x.shape)
            x = layer(x)
        return x.T
