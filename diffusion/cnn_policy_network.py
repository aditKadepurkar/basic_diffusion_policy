"""
This file will contain the CNN based policy 
architecture proposed in the original 
Diffusion Policy paper.

https://arxiv.org/pdf/2303.04137v4
"""

# import os
# import sys
# sys.path.append(os.getcwd())

import equinox as eqx
import jax
from diffusion.embedding_layer import SinusoidalPositionEmbedding

class CnnDiffusionPolicy(eqx.Module):
    def __init__(self, action_dim, obs_dim):
        key = jax.random.PRNGKey(15)

        key0, key1, key2, key3 = jax.random.split(key, 4)


        dims = [action_dim, 256, 512, 1024]
        kernel_size = 5
        embed_dim = 256
        n_groups = 8

        encoder = eqx.nn.Sequential(layers=[
                SinusoidalPositionEmbedding(embed_dim),
                eqx.nn.Linear(embed_dim, embed_dim * 4, key=key0),
                jax.nn.mish,
                eqx.nn.Linear(embed_dim * 4, embed_dim, key=key1),
        ]
        )

        total_dim = obs_dim + embed_dim

        # print(list(zip(dims[:-1], dims[1:])))

        mid = eqx.nn.Sequential(layers=[
            eqx.nn.Conv1D(total_dim, dims[1], kernel_size=kernel_size, key=key2),
            jax.nn.mish,
            eqx.nn.Conv1D(dims[1], dims[2], kernel_size=kernel_size, key=key3),
            jax.nn.mish,
        ])






        # self.layers = [
        # ]


    def __call__(x):
        pass


