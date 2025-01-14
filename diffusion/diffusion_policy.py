import jax.numpy as jnp
import equinox as eqx
import jax
from diffusion.data_loader import DataLoader
import optax
from diffusion.mlp_model import MLP
import math

def gamma_to_alpha_sigma(gamma, scale = 1):
    return jnp.sqrt(gamma) * scale, jnp.sqrt(1 - gamma)

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return jnp.float16(output).clip(min = clip_min)



class NoiseScheduler:

    def __init__(self, T, beta_schedule):
        self.betas = beta_schedule
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        assert len(self.betas) == T

    def add_noise(
        self,
        original_samples,
        noise,
        timesteps,
    ):
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod
        timesteps = timesteps

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()


        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = jnp.expand_dims(sqrt_alpha_prod, -1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = jnp.expand_dims(sqrt_one_minus_alpha_prod, -1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples




    def step(
        self, noise_pred, t, action
    ):
        prev_t = t - 1

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else jnp.array(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t


        pred_original_sample = (action - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * action

        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = jnp.clip(variance, a_min=1e-20)

        variance = 0
        if t > 0:
            device = noise_pred.device
            variance_noise = jax.random.normal(jax.random.PRNGKey(t), noise_pred.shape)

            variance = (variance ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        return (pred_prev_sample,)














class DiffusionPolicy:
    def __init__(self, key, data_path, T=50, gamma=0.99, sigma=0.2):
        # Initialize data loader and optimizer
        if data_path is None:
            self.data_loader = None
        else:
            self.data_loader = DataLoader(data_path, "data", 32)

        self.alpha = jnp.cos(jnp.linspace(0, jnp.pi / 2, T)) ** 2
        self.sigma = sigma
        self.gamma = cosine_schedule

        self.key = key
        # Initialize model
        # Policy network should be initialized here
        model = MLP(40, 7)
        # params = eqx.filter(model, eqx.is_inexact_array)
        # print(f"Initial Params: {params}")
        self.model = model

    def inference(x, model, key, T=50, n_actions=4, output_dim=7):
        # Forward pass through the 
        # Should simplify the code.
        # x will be the observations and states of the expert
        
        # gamma = cosine_schedule(1)

        # _, sigma = gamma_to_alpha_sigma(gamma)

        # alpha = jnp.cos(jnp.linspace(0, jnp.pi / 2, T)) ** 2

        key, subkey = jax.random.split(key)
        a_t = jax.random.normal(subkey, (x.shape[0], n_actions, output_dim))

        for k in range(T):
            # gamma = cosine_schedule(k)
            # _, sigma = gamma_to_alpha_sigma(gamma)

            a_t = DiffusionPolicy.diffusion_step(a_t, x, model, jnp.array(k))
            key, subkey = jax.random.split(key)
            a_t += jax.random.normal(subkey, (x.shape[0], n_actions, output_dim))
            # a_t = a_t

        return a_t


    # @jax.jit
    def predict_action(model, observation, Y, T, key, n_actions, output_dim=7, gamma=0.8, alpha=[0.99], sigma=5):
        # Generate initial random actions
        key, subkey = jax.random.split(key)

        gamma = cosine_schedule(1)

        alpha = jnp.cos(jnp.linspace(0, jnp.pi / 2, T)) ** 2

        _, sigma = gamma_to_alpha_sigma(gamma)

        a_t = sigma * jax.random.normal(subkey, (observation.shape[0], n_actions, output_dim))


        cache_X = []
        cache_Y = DiffusionPolicy.forward_diffusion(Y, T)
        
        # Perform T diffusion steps
        for k in range(T):
            gamma = cosine_schedule(k)
            _, sigma = gamma_to_alpha_sigma(gamma)

            a_t = DiffusionPolicy.diffusion_step(a_t, observation, model, jnp.array(k), gamma)
            key, subkey = jax.random.split(key)
            a_t += sigma * jax.random.normal(subkey, (observation.shape[0], n_actions, output_dim))
            a_t = alpha[T-k-1] * a_t
            cache_X.append(a_t)


        # Calculate loss and gradients
        loss_value, grads = eqx.filter_value_and_grad(DiffusionPolicy.loss)(model, cache_Y, cache_X, observation) # (cache_Y, cache_X)

        return a_t, loss_value, grads

    # @jax.jit
    def diffusion_step(a_t, observation, model, k, gamma):
        batch, n_actions, action_dim = a_t.shape
        k = jnp.full((batch, n_actions, 1), k)
        
        # observation = jnp.broadcast_to(observation, (observation.shape[0], n_actions, observation.shape[2]))

        # print(a_t.shape, observation.shape, k.shape)

        nn_input = jnp.concatenate([a_t, observation, k], axis=2)

        # print(nn_input.shape)
        a_t1 = jax.vmap(model)(nn_input)


        a_t1 = a_t - a_t1

        return a_t1


    # @jax.jit
    def forward_diffusion(a_0, T=50):
        a_t = jnp.copy(a_0)
        alpha_schedule = jnp.cos(jnp.linspace(0, jnp.pi / 2, T)) ** 2

        cache_Y = []

        key = jax.random.PRNGKey(42)

        for t in range(T):
            alpha = alpha_schedule[t]
            key, subkey = jax.random.split(key)
            epsilon = jax.random.normal(subkey, a_0.shape)
            a_t = jnp.sqrt(alpha) * a_0 + jnp.sqrt(1 - alpha) * epsilon
            
            cache_Y.insert(0, a_t)

        return cache_Y

    # @jax.jit
    def loss(model, Y, X, observation, gamma = 0.8):
        # Model predictions using vmap
        
        # Y = jnp.array(Y)
        X = jnp.array(X)

        # print(X.shape)

        pred_y = [0] * len(X)

        steps, batch, n_actions, action_dim = X.shape

        k = jnp.arange(steps)
        k = k.reshape((steps, 1, 1, 1))
        k = jnp.broadcast_to(k, (steps, batch, n_actions, 1))

        observation = jnp.tile(jnp.expand_dims(observation, axis=0), (steps, 1, 1, 1))

        # print(X.shape, observation.shape, k.shape)


        nn_input = jnp.concatenate([X, observation, k], axis=3)


        batched_model = jax.vmap(model)

        pred_y = jax.vmap(batched_model)(nn_input)

        # for i in range(len(X)):
        #     nn_input = jnp.concatenate([X[i], observation, jnp.full((batch, n_actions, 1), i)], axis=2)
        #     pred_y[i] = jax.vmap(model)(nn_input) - gamma * X[i]

        # pred_y = jax.vmap(DiffusionPolicy.diffusion_step)(X, observation, model, jnp.arange(len(X)), jnp.full(len(X), 0.8))




        return DiffusionPolicy.MSE(Y, pred_y)

    # @jax.jit
    def MSE(y, X):
        # print(type(y), type(X))
        v = (jnp.array(X) - jnp.array(y)) ** 2
        print(v)
        print(jnp.max(v).item())
        print(jnp.min(v).item())
        exit(0)
        return jnp.mean((jnp.array(X) - jnp.array(y)) ** 2)

