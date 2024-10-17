import h5py
import jax.numpy as jnp
import jax

# Utility function to shuffle two arrays in unison
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    key = jax.random.PRNGKey(0)
    p = jax.random.permutation(key, len(a))
    return a[p], b[p]

class DataLoader():
    def __init__(self, filename, batch_size=4, shuffle=True):
        self.filename = filename
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = 0
        
        with h5py.File(self.filename, "r") as f:
            self.data_keys = list(f['data'].keys())

    def load_data_in_batches(self):
        with h5py.File(self.filename, "r") as f:
            data = f['data']

            for key in self.data_keys:
                # Extract states and actions
                actions = data[key]['actions']
                states = data[key]['states']

                num_samples = len(states) - 4
                indices = jnp.arange(0, num_samples, 4)

                if self.shuffle:
                    key = jax.random.PRNGKey(0)
                    indices = jax.random.permutation(key, indices)

                for i in range(0, len(indices), self.batch_size):
                    batch_indices = indices[i:i + self.batch_size]

                    batch_states = [states[idx:idx + 4] for idx in batch_indices]
                    batch_actions = [actions[idx:idx + 4] for idx in batch_indices]

                    batch_states = jnp.array(batch_states)
                    batch_actions = jnp.array(batch_actions)

                    if self.shuffle:
                        batch_states, batch_actions = unison_shuffled_copies(batch_states, batch_actions)

                    yield {'states': batch_states, 'actions': batch_actions}

# Example usage:
# loader = DataLoader('your_file.hdf5', batch_size=4)
# for batch in loader.load_data_in_batches():
#     # process batch['states'] and batch['actions']


# loader = DataLoader('/home/aditkadepurkar/dev/diffusiontest/data/1728922451_627212/demo.hdf5', batch_size=4)

# for batch in loader.load_data_in_batches():
#     print(batch['states'].shape, batch['actions'].shape)
