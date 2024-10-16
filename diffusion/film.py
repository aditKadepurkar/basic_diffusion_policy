"""
Implementation of Feature-wise Linear Modulation

https://arxiv.org/pdf/1709.07871
"""
import equinox as eqx


class FiLM(eqx.Module):
    def __init__(self, in_channels, out_channels, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key = key
        
        
        

    def __call__(self, x):
        pass
    
    
    
class FiLMBlock(eqx.Module):
    """
    This class builds a block with a FiLM layer, and a convolutional layer
    
    
    
    """
    
    def __init__(self):
        
        
        
        self.layers = [
            FiLM(),
            eqx.nn.Conv3d(),
        ]
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        

