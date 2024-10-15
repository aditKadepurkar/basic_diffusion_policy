"""
This file contains the diffusion policy class that will be used to train the policy
"""

import jax.numpy as jnp
import equinox as eqx
import jax



class DiffusionPolicy():
    def __init__(self, key):
        # start with a basic nn policy

        self.key = key

        self.key, key1, key2, key3 = jax.random.split(self.key, 4)

        self.layer1 = eqx.nn.Linear(15, 64, key=key1)
        self.activation1 = eqx.nn.PReLU()
        self.layer2 = eqx.nn.Linear(64, 64, key=key2)
        self.activation2 = eqx.nn.PReLU()
        self.layer3 = eqx.nn.Linear(64, 7, key=key3)

    def train(self, demonstrations):
        pass

    def save(self, output_dir):
        pass

    def forward(self, x):
        # x shape: (action_dim + observation_dim + 1,)
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        return x

    def predict_action(self, observation, T, n_actions, output_dim = 7):
        self.key, subkey = jax.random.split(self.key)
        a_t = jax.random.normal(self.key, (n_actions, output_dim))
        
        for k in range(T):
            a_t = self.diffusion_step(a_t, observation, jnp.array(k))
        
        return a_t

    def diffusion_step(self, a_t, observation, k):
        """
        Perform one step of the diffusion process
        """

        n_actions, action_dim = a_t.shape
        
        k = jnp.full((n_actions, 1), k)
        
        observation = jnp.broadcast_to(observation[None, :], (n_actions, observation.shape[0]))
        
        nn_input = jnp.concatenate([a_t, observation, k], axis=1)
        
        a_t1 = jax.vmap(self.forward)(nn_input)
        
        return a_t1

        # I think generally I will put the observation through a CNN first?
        # Or I could go with the transformer approach
        # For now lets assume all of these are just 1 dimensional


        # the observation and k should be concatenated to each action in a_t


    def forward_diffusion(self, a_0, T):

        a_t = jnp.copy(a_0)

        alpha_schedule = jnp.cos(jnp.linspace(0, jnp.pi / 2, T)) ** 2

        for t in range(T):
            alpha = alpha_schedule[t]
            key, subkey = jax.random.split(self.key)
            epsilon = jax.random.normal(subkey, a_0.shape)

            a_t = jnp.sqrt(alpha) * a_0 + jnp.sqrt(1 - alpha) * epsilon
        
        return a_t

