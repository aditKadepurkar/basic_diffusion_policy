import jax.numpy as jnp
import equinox as eqx
import jax
from diffusion.data_loader import DataLoader
import optax
from diffusion.mlp_model import MLP



class DiffusionPolicy:
    def __init__(self, key, data_path):
        # Initialize data loader and optimizer
        self.data_loader = DataLoader(data_path)

        self.key = key
        # Initialize model
        # Policy network should be initialized here
        model = MLP(46, 7)
        # params = eqx.filter(model, eqx.is_inexact_array)
        # print(f"Initial Params: {params}")
        self.model = model



    # @jax.jit
    def predict_action(model, observation, Y, T, key, n_actions, output_dim=7):
        # Generate initial random actions
        key, subkey = jax.random.split(key)
        a_t = jax.random.normal(subkey, (n_actions, output_dim))

        cache_X = []
        cache_Y = DiffusionPolicy.forward_diffusion(Y, T)
        
        # Perform T diffusion steps
        for k in range(T):
            a_t = DiffusionPolicy.diffusion_step(model, a_t, observation, jnp.array(k))
            cache_X.append(a_t)


        # Calculate loss and gradients
        loss_value, grads = eqx.filter_value_and_grad(DiffusionPolicy.loss)(model, cache_Y, cache_X) # (cache_Y, cache_X)

        return a_t, loss_value, grads

    # @jax.jit
    def diffusion_step(model, a_t, observation, k):
        n_actions, action_dim = a_t.shape
        k = jnp.full((n_actions, 1), k)
        observation = jnp.broadcast_to(observation[None, :], (n_actions, observation.shape[0]))
        nn_input = jnp.concatenate([a_t, observation, k], axis=1)
        a_t1 = jax.vmap(model)(nn_input)
        return a_t1

    # @jax.jit
    def forward_diffusion(a_0, T=1000):
        a_t = jnp.copy(a_0)
        alpha_schedule = jnp.cos(jnp.linspace(0, jnp.pi / 2, T)) ** 2

        cache_Y = []

        key = jax.random.PRNGKey(0)

        for t in range(T):
            alpha = alpha_schedule[t]
            key, subkey = jax.random.split(key)
            epsilon = jax.random.normal(subkey, a_0.shape)
            a_t = jnp.sqrt(alpha) * a_0 + jnp.sqrt(1 - alpha) * epsilon
            
            cache_Y.insert(0, a_t)

        return cache_Y

    # @jax.jit
    def loss(model, Y, X):
        # Model predictions using vmap
        
        # Y = jnp.array(Y)
        # X = jnp.array(X)

        # pred_y = [0] * len(X)

        # for i in range(len(X)):
        #     pred_y[i] = jnp.array(X[i])

        # pred_y = jax.vmap(model)(X)

        return DiffusionPolicy.MSE(Y, X)

    @jax.jit
    def MSE(y, X):
        # print(type(y), type(X))
        return jnp.mean((jnp.array(X) - jnp.array(y)) ** 2)

