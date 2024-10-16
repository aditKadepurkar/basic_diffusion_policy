

import equinox as eqx

class EmbeddingLayer(eqx.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels

    def __call__(self, x):
        pass

