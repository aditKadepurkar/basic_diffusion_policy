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
from diffusion.embedding_layer import embedding_layer, mish

class CnnDiffusionPolicy(eqx.Module):
    layers: list
    def __init__(self, action_dim, obs_dim):
        key = jax.random.PRNGKey(15)

        key0, key1, key2, key3 = jax.random.split(key, 4)
        del key


        dims = [action_dim, 256, 512, 1024]
        kernel_size = 5
        embed_dim = 256
        n_groups = 8

        encoder = [
                embedding_layer,
                eqx.nn.Linear(embed_dim, embed_dim * 4, key=key0, use_bias=False),
                mish,
                eqx.nn.Linear(embed_dim * 4, embed_dim, key=key1, use_bias=False),
        ]
        
        del key0, key1

        total_dim = obs_dim + embed_dim

        dim_pairs = list(zip(dims[:-1], dims[1:]))
        
        key0, key1 = jax.random.split(key2, 2)
        del key2

        mid = eqx.nn.Sequential(layers=[
            ResidualBlock(
                dims[-1], dims[-1], cond_dim=total_dim,
                kernel_size=kernel_size, n_groups=n_groups, key=key0
            ),
            ResidualBlock(
                dims[-1], dims[-1], cond_dim=total_dim,
                kernel_size=kernel_size, n_groups=n_groups, key=key1
            ),
        ])
        
        del key0, key1
        

        down_modules = []
        for ind, (dim_in, dim_out) in enumerate(dim_pairs):
            key0, key1, key2, key3 = jax.random.split(key3, 4)
            is_last = ind >= (len(dim_pairs) - 1)
            down_modules.append([
                ResidualBlock(
                    dim_in, dim_out, cond_dim=total_dim,
                    kernel_size=kernel_size, n_groups=n_groups, key=key0),
                ResidualBlock(
                    dim_out, dim_out, cond_dim=total_dim,
                    kernel_size=kernel_size, n_groups=n_groups, key=key1),
                eqx.nn.Conv1d(dim_out, dim_out, 3, 2, 1, key=key2) if not is_last else eqx.nn.Identity()
            ])
            del key0, key1, key2


        up_modules = []
        for ind, (dim_in, dim_out) in enumerate(reversed(dim_pairs)):
            key0, key1, key2, key3 = jax.random.split(key3, 4)
            is_last = ind >= (len(dim_pairs) - 1)
            up_modules.append([
                ResidualBlock(
                    dim_out * 2, dim_in, cond_dim=total_dim,
                    kernel_size=kernel_size, n_groups=n_groups, key=key0),
                ResidualBlock(
                    dim_in, dim_in, cond_dim=total_dim,
                    kernel_size=kernel_size, n_groups=n_groups, key=key1),
                eqx.nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1, key=key2) if not is_last else eqx.nn.Identity()
            ])
            del key0, key1, key2

        key0, key1 = jax.random.split(key3, 2)
        del key3        

        final_conv = eqx.nn.Sequential(layers=[
            Conv1dBlock(dims[0], dims[0], kernel_size=kernel_size, key=key0),
            eqx.nn.Conv1d(dims[0], action_dim, 1, key=key1),
        ])


        self.layers = [
            encoder,
            down_modules,
            mid,
            up_modules,
            final_conv
        ]


    def __call__(self, x, timestep, cond):
        # print(x.shape, timestep.shape, cond.shape)

        for layer in self.layers[0]:
            # print(timestep.shape)
            timestep = layer(timestep.T)

        features = timestep.T

        # resize x
        # x = jnp.array([x])
        # x = jnp.reshape(x, (1, -1))

        # print("At 115, ", x.shape, features.shape, cond.shape)

        # encoder
        features = jnp.concatenate([features, cond], axis=-1)

        # down modules
        h = []
        for (conv1, conv2, down) in self.layers[1]:
            x = conv1(x, features)
            x = conv2(x, features)
            h.append(x)
            x = down(x)

        # mid
        for residual in self.layers[2]:
            x = residual(x, features)

        # up modules
        for (conv1, conv2, up) in self.layers[3]:
            x = jnp.concatenate([x, h.pop()], axis=-1)
            x = conv1(x, features)
            x = conv2(x, features)
            x = up(x)

        # final conv
        x = self.layers[4](x)

        return x

class ResidualBlock(eqx.Module):
    layers: list
    out_channels: int
    def __init__(self, dim_in, dim_out, cond_dim, kernel_size, n_groups, key):
        subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 4)
        del key
        cond_channels = dim_out * 2
        out_channels = dim_out
        cond_encoder = [
            mish,
            eqx.nn.Linear(cond_dim, cond_channels, key=subkey1, use_bias=False),
        ]

        residual = eqx.nn.Conv1d(dim_in, out_channels, 1, key=subkey2) \
            if dim_in != out_channels else eqx.nn.Identity()


        self.layers = [
            Conv1dBlock(dim_in, dim_out, subkey3, kernel_size, n_groups),
            cond_encoder,
            Conv1dBlock(dim_in, dim_out, subkey4, kernel_size, n_groups),
            residual,
        ]

        self.out_channels = out_channels

    def __call__(self, x, obs, key=None):
        # print(x.shape, obs.shape)

        out = self.layers[0](x)

        # print(out.shape)

        embedding = obs.T
        # encoder
        for layer in self.layers[1]:
            embedding = layer(embedding)
        embedding = embedding


        # print(embedding.shape)

        # reshape embedding
        embed = embedding.reshape(embedding.shape[0]//2, 2, -1)
        
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
                
        out = out * scale + bias

        out = self.layers[2](out)

        out = out[:28, :]

        print(out.shape)


        return out + self.layers[3](x)


class Conv1dBlock(eqx.Module):
    layers: list
    def __init__(self, inp_channels, out_channels, key, kernel_size, n_groups=8):

        if out_channels == 28:
            n_groups = 7
        self.layers = [
            eqx.nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2, key=key, use_bias=False),
            eqx.nn.GroupNorm(n_groups, out_channels),
            jax.nn.mish,
        ]

    def __call__(self, x):
        x = x.T
        for layer in self.layers:
            # print(x.shape)
            x = layer(x)
        return x.T
