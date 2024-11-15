import robosuite as suite
from robosuite.controllers import load_controller_config
import numpy as np
from diffusion.cnn_policy_network import get_vision_encoder
import jax

resnet = get_vision_encoder()

x = np.zeros((3, 64, 64))

print(x.shape)
out = resnet(x=x, key=jax.random.PRNGKey(0))
print(out.shape)

