import jax.numpy as jnp
import equinox as eqx
import jax
from diffusion.data_loader import DataLoader
import optax
from diffusion.mlp_model import MLP


class DiffusionPolicy:
    def __init__(self, key, data_path, lr=1e-3):
        # Initialize data loader and optimizer
        self.data_loader = DataLoader(data_path)
        self.optim = optax.adamw(learning_rate=lr)
        self.key = key
        # Initialize model
        # Policy network should be initialized here
        model = MLP(46, 7)
        # params = eqx.filter(model, eqx.is_inexact_array)
        # print(f"Initial Params: {params}")
        self.model = model


    def train(self):
        model = self.model
        params = eqx.filter(model, eqx.is_inexact_array)
        # print(f"Initial Params: {params}")
        opt_state = self.optim.init(params)

        # Load data
        data = self.data_loader.load_data(count=1)
        for demo in data:
            observations = data[demo]['states']
            expert_action_sequence = data[demo]['actions']

            # Loop over each observation
            loss_period = 0
            for i in range(len(observations)):
                a_t, loss_value, grads = self.predict_action(
                    model=model,
                    observation=jnp.copy(observations[i]),
                    Y=jnp.array(expert_action_sequence[i]),
                    T=1000,
                    n_actions=4
                )
                loss_period += loss_value
                if (i + 1) % 4 == 0:
                    loss_period /= 4
                    print(f"Loss: {loss_period}")
                # print(f"Gradients: {grads}")

                    params = eqx.filter(model, eqx.is_array)
                    updates, opt_state = self.optim.update(grads, opt_state, params)
                    model = eqx.apply_updates(model, updates)

                # print(f"Updated Params: {params}")



    # @jax.jit
    def predict_action(model, observation, Y, T, key, n_actions, output_dim=7):
        # Generate initial random actions
        key, subkey = jax.random.split(key)
        a_t = jax.random.normal(subkey, (n_actions, output_dim))

        # Perform T-1 diffusion steps
        for k in range(T - 1):
            a_t = DiffusionPolicy.diffusion_step(model, a_t, observation, jnp.array(k))

        # Final diffusion step
        k = jnp.full((n_actions, 1), T - 1)
        observation = jnp.broadcast_to(observation[None, :], (n_actions, observation.shape[0]))
        nn_input = jnp.concatenate([a_t, observation, k], axis=1)

        # Calculate loss and gradients
        loss_value, grads = eqx.filter_value_and_grad(DiffusionPolicy.loss)(model, nn_input, Y)

        return a_t, loss_value, grads

    # @jax.jit
    def diffusion_step(model, a_t, observation, k):
        n_actions, action_dim = a_t.shape
        k = jnp.full((n_actions, 1), k)
        observation = jnp.broadcast_to(observation[None, :], (n_actions, observation.shape[0]))
        nn_input = jnp.concatenate([a_t, observation, k], axis=1)
        a_t1 = jax.vmap(model)(nn_input)
        return a_t1

    def forward_diffusion(self, a_0, T):
        a_t = jnp.copy(a_0)
        alpha_schedule = jnp.cos(jnp.linspace(0, jnp.pi / 2, T)) ** 2

        for t in range(T):
            alpha = alpha_schedule[t]
            self.key, subkey = jax.random.split(self.key)
            epsilon = jax.random.normal(subkey, a_0.shape)
            a_t = jnp.sqrt(alpha) * a_0 + jnp.sqrt(1 - alpha) * epsilon
        
        return a_t

    # @jax.jit
    def loss(model, a_t, Y):
        # Model predictions using vmap
        pred_y = jax.vmap(model)(a_t)
        return DiffusionPolicy.MSE(Y, pred_y)

    @jax.jit
    def MSE(y, pred_y):
        return jnp.mean((pred_y - y) ** 2)

