"""
This file will contain the CNN based policy 
architecture proposed in the original 
Diffusion Policy paper.
"""

import equinox as eqx
import jax
from eqxvision.models import resnet18
from diffusion.film import FiLM
from diffusion.embedding_layer import EmbeddingLayer

class CnnDiffusionPolicy(eqx.Module):
    def __init__(self):
        key = jax.random.PRNGKey(0)

        self.layers = [
            FiLM(3, 64, key=key),
            eqx.nn.Conv2D(64, 64, 3, 1, 1),
            EmbeddingLayer(64, 64),
            FiLM(3, 64, key=key),
            eqx.nn.Conv2D(64, 64, 3, 1, 1),
            EmbeddingLayer(64, 64),
        ]



        

    def __call__(x):
        pass



