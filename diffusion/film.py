
import equinox as eqx


class FiLM(eqx.Module):
    def __init__(self, in_channels, out_channels, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key = key

    def __call__(self, x):
        pass