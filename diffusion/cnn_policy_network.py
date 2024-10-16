"""
This file will contain the CNN based policy 
architecture proposed in the original 
Diffusion Policy paper.

https://arxiv.org/pdf/2303.04137v4
"""

import equinox as eqx
import jax
from eqxvision.models import resnet18
from diffusion.film import FiLMBlock
from diffusion.embedding_layer import EmbeddingLayer

class CnnDiffusionPolicy(eqx.Module):
    def __init__(self):
        key = jax.random.PRNGKey(0)

        self.layers = [
            FiLMBlock(3, 64, key=key),
            eqx.nn.Conv2D(64, 64, 3, 1, 1),
            EmbeddingLayer(64, 64),
            FiLMBlock(3, 64, key=key),
            eqx.nn.Conv2D(64, 64, 3, 1, 1),
            EmbeddingLayer(64, 64),
        ]



        

    def __call__(x):
        pass



