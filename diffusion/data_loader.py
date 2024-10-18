"""
Dataloader for loading data from HDF5 file.
"""

import h5py
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import time

class DataLoader():
    file_path: str
    dataset_name: str
    batch_size: int
    shuffle: bool
    buffer_size: int = 1000
    _indices: np.ndarray = eqx.static_field()

    def __init__(self, file_path, dataset_name, batch_size, shuffle=True, buffer_size=1000):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self._dataset_size = 0
        file = h5py.File(file_path, 'r')

        self.labels = file[dataset_name]['demo_1']['actions']

        self.sizes = []

        with h5py.File(file_path, 'r') as f:
            # print(f[dataset_name].keys())
            for demo in f[dataset_name].keys():
                self.sizes.append(f[dataset_name][demo]['states'].shape[0] - 8 + self.sizes[-1] if len(self.sizes) > 0 else f[dataset_name][demo]['states'].shape[0] - 8)
                self._dataset_size += f[dataset_name][demo]['states'].shape[0] -8
            # print(self.sizes)
            self._dataset_size = f[dataset_name]['demo_1']['states'].shape[0]

        self._indices = np.arange(self._dataset_size - 8)
        if self.shuffle:
            self._indices = jax.random.permutation(jax.random.PRNGKey(0), self._indices)
            # np.random.shuffle(self._indices)

    def _load_data(self, indices):
        """Loads the required data from HDF5 file based on indices."""

        indices = sorted(indices)
        with h5py.File(self.file_path, 'r') as f:
            data = f[self.dataset_name]['demo_1']
            states = []
            actions = []
            states.append([data['states'][i:i+4] for i in indices])
            actions.append([data['actions'][i+4:i+8] for i in indices])
            data = {"states": jnp.array(states), "actions": jnp.array(actions)}
        return data

    def __iter__(self):
        self._start = 0
        return self

    def __next__(self):
        if self._start >= self._dataset_size:
            raise StopIteration

        end = min(self._start + self.batch_size, self._dataset_size)
        batch_indices = self._indices[self._start:end]
        self._start = end

        # Load and return batch
        return self._load_data(batch_indices)

    def shuffle_data(self):
        """Shuffle the indices if needed."""
        if self.shuffle:
            self._indices = jax.random.permutation(jax.random.PRNGKey(0), self._indices)

# Usage
# file_path = "/home/aditkadepurkar/dev/diffusiontest/data/1728922451_627212/demo.hdf5"
# dataset_name = "data"
# batch_size = 32

# data_loader = DataLoader(file_path, dataset_name, batch_size, shuffle=True)

# # Iterate over batches
# for batch in data_loader:
    
    # Your training loop or processing code here
    # time.sleep(60)
    # print(batch.shape)


# data_loader = DataLoader(file_path, dataset_name, batch_size, shuffle=True)

# # Iterate over batches
# for batch in data_loader:
    # Your training loop or processing code here
    # time.sleep(60)
    # print(batch.shape)
