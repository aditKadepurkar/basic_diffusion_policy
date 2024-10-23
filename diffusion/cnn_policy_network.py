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
import jax.numpy as jnp
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

        dim_pairs = list(zip(dims[:-1], dims[1:]))

        mid = eqx.nn.Sequential(layers=[
            ResidualBlock(
                dims[-1], dims[-1], cond_dim=total_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ResidualBlock(
                dims[-1], dims[-1], cond_dim=total_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = []
        for ind, (dim_in, dim_out) in enumerate(dim_pairs):
            is_last = ind >= (len(dim_pairs) - 1)
            down_modules.append([
                ResidualBlock(
                    dim_in, dim_out, cond_dim=total_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ResidualBlock(
                    dim_out, dim_out, cond_dim=total_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                eqx.nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else eqx.nn.Identity()
            ])

        up_modules = []
        for ind, (dim_in, dim_out) in enumerate(reversed(dim_pairs)):
            is_last = ind >= (len(dim_pairs) - 1)
            up_modules.append([
                ResidualBlock(
                    dim_out * 2, dim_in, cond_dim=total_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ResidualBlock(
                    dim_in, dim_in, cond_dim=total_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                eqx.nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1) if not is_last else eqx.nn.Identity()
            ])

        final_conv = eqx.nn.Sequential(layers=[
            Conv1dBlock(dims[0], dims[0], kernel_size=kernel_size),
            eqx.nn.Conv1d(dims[0], action_dim, 1)
        ])

        self.layers = [
            encoder,
            mid,
            down_modules,
            up_modules,
            final_conv
        ]

    def __call__(self, x, timestep, cond):
        features = self.layers[0](timestep)

        features = jnp.concatenate([features, cond], axis=-1)
        
        for layer in self.layers:
            x = layer(x)

class ResidualBlock(eqx.Module):
    def __init__(self, dim_in, dim_out, cond_dim, kernel_size, n_groups):

        cond_channels = dim_out * 2
        out_channels = dim_out
        cond_encoder = eqx.nn.Sequential(layers=[
            jax.nn.mish,
            eqx.nn.Linear(cond_dim, cond_channels),
            eqx.nn.Unflatten(-1, (-1, 1))
        ])

        residual = eqx.nn.Conv1d(dim_in, out_channels, 1) \
            if dim_in != out_channels else eqx.nn.Identity()


        self.layers = [
            Conv1dBlock(dim_in, dim_out, kernel_size, n_groups),
            cond_encoder,
            Conv1dBlock(dim_in, dim_out, kernel_size, n_groups),
            residual,
        ]

        self.out_channels = out_channels

    def __call__(self, x):
        # TODO update this

        out = self.layers[0](x)
        embedding = self.layers[1](x)

        embed = embedding.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = out * scale + bias

        out = self.layers[2](out)

        return out + self.layers[3](x)


class Conv1dBlock(eqx.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):

        self.layers = [
            eqx.nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            eqx.nn.GroupNorm(n_groups, out_channels),
            jax.nn.mish,
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
