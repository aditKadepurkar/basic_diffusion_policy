import jax.numpy as jnp
import equinox as eqx
import jax

# Example input with shape (256, 28, 1)
input_tensor = jax.random.normal(key=jax.random.PRNGKey(11), shape=(49, 4))

# Initialize Conv1d layer
inp_channels = 49  # Input channels must match the last dimension of reshaped_input
out_channels = 49  # Define number of output channels to match the desired output shape
kernel_size = 5  # Define kernel size
key = jax.random.PRNGKey(1)  # Initialize your random key

conv_layer = eqx.nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2, key=key)

print(conv_layer)

# Call the Conv1d layer
output = conv_layer(input_tensor)  # Add a channel dimension

print("Output shape:", output.shape)  # This should print the output shape of the Conv1d layer
