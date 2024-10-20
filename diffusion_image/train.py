import sys
import os
sys.path.append(os.getcwd())

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from image_data_loader import DataLoader
from diffusion.diffusion_policy import DiffusionPolicy

"""
Trains on images using JAX and Equinox
"""

# Load data
train_loader = DataLoader('microsoft/cats_vs_dogs')

# Initialize model, loss function, and optimizer
agent = DiffusionPolicy(key=jax.random.PRNGKey(0), data_path=None)
criterion = optax.softmax_cross_entropy
optimizer = optax.adam(0.001)

model = agent.model

# Initialize optimizer state
opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

# Training loop
num_epochs = 10

@jax.jit
def train_step(agent, opt_state, images, labels):
    a_t, loss_value, grads = DiffusionPolicy.predict_action(
        model=model,
        observation=labels,
        Y=images,
    )

    # grads, _ = jax.grad(loss_fn, has_aux=True)(agent)
    updates, opt_state = optimizer.update(grads, opt_state)
    agent = eqx.apply_updates(agent, updates)
    return agent, opt_state, loss_value

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        agent, opt_state, loss = train_step(agent, opt_state, images, labels)
        outputs = agent(images)
        # loss = jnp.mean(criterion(outputs, labels))
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # Validation loop (if needed)
    # val_loss = 0.0
    # for images, labels in val_loader:
    #     outputs = agent(images)
    #     loss = jnp.mean(criterion(outputs, labels))
    #     val_loss += loss.item()
    
    # print(f"Validation Loss: {val_loss/len(val_loader)}")