

import equinox as eqx
import jax
import jax.numpy as jnp
import math

# class SinusoidalPositionEmbedding(eqx.Module):
#     embed_dim: int
    # def __init__(self, embed_dim):
    #     self.embed_dim = embed_dim

def embedding_layer(x):
    embed_dim = 256
    embed = math.log(10000) / (embed_dim // 2 - 1)
    embed = jnp.exp(jnp.arange(embed_dim // 2) * -embed)
    embed = jnp.expand_dims(x, 0) * embed
    embed = jnp.concatenate([jnp.sin(embed), jnp.cos(embed)], axis=0)
    return embed


