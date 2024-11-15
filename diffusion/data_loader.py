"""
Dataloader for loading data from HDF5 file.
"""

import h5py
import numpy as np
# import jax
# import jax.numpy as jnp
import numpy as jnp
# import equinox as eqx
import time

class DataLoader():
    file_path: str
    dataset_name: str
    batch_size: int
    shuffle: bool
    buffer_size: int = 1000

    def __init__(self, file_path, dataset_name, batch_size, shuffle=True, buffer_size=1000):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self._dataset_size = 0
        file = h5py.File(file_path, 'r')
        
        self.stats = list(get_data_stats_head(file))
        # print(self.stats)

        self.labels = file[dataset_name]['demo_1']['actions']

        self.sizes = []
        self.sets = []

        with h5py.File(file_path, 'r') as f:
            # print(f[dataset_name].keys())
            for demo in f[dataset_name].keys():
                self.sizes.append(f[dataset_name][demo]['states'].shape[0] - 9 + self.sizes[-1] if len(self.sizes) > 0 else f[dataset_name][demo]['states'].shape[0] - 9)
                self._dataset_size += f[dataset_name][demo]['states'].shape[0] -9
                self.sets.append(demo)
        
        del f
        
        self.sizes = jnp.array(self.sizes)
        # self.sets = jnp.array(self.sets)

            # print(self.sizes)
            # self._dataset_size = f[dataset_name]['demo_1']['states'].shape[0]

        print(f"Expert demos: {len(self.sets)}")

        self._indices = np.arange(self._dataset_size)
        self._indices = jnp.concatenate([self._indices, self._indices, self._indices])
        if self.shuffle:
            self._indices = np.random.permutation(self._indices)
            # np.random.shuffle(self._indices)
        

    def _load_data(self, indices):
        """Loads the required data from HDF5 file based on indices."""

        indices = sorted(indices)
        states = []
        actions = []
        with h5py.File(self.file_path, 'r') as f:
            for i in indices:
                idx = jnp.searchsorted(self.sizes, i, side='right')
                if i == self.sizes[idx]:
                    # the first state of a demo

                    # concat first state 4 times
                    # data['states'][i]

                    data = f[self.dataset_name][self.sets[idx]]

                    states.append(jnp.array([data['states'][0]] * 2))
                    actions.append(data['actions'][0:8])

                else:
                    if idx > 0:
                        i -= self.sizes[idx - 1]
                    

                    demo = self.sets[idx]

                    data = f[self.dataset_name][demo]

                    states.append(jnp.ravel(data['states'][i:i+2]))
                    actions.append(data['actions'][i+1:i+9])

        actions = DataLoader.normalize_data_actions(jnp.array(actions, dtype=jnp.float16), self.stats[0])
        states = DataLoader.normalize_data(jnp.array(states, dtype=jnp.float16), self.stats[1])
        
        data = {"states": states, "actions": actions}
            # print(data['states'].shape, data['actions'].shape)
        del f
        del states, actions
        return data

    def __iter__(self):
        self._start = 0
        return self

    def __next__(self):
        if self._start >= len(self._indices):
            raise StopIteration

        end = min(self._start + self.batch_size, len(self._indices))
        batch_indices = self._indices[self._start:end]
        self._start = end

        # Load and return batch
        return self._load_data(batch_indices)
    

    def shuffle_data(self):
        """Shuffle the indices if needed."""
        if self.shuffle:
            self._indices = np.random.permutation(self._indices)
            # print(self._indices)
            # self._indices = self._indices[:int(0.7*len(self._indices))]
            # print(self._indices)
    def get_batch_count(self):
        return len(self._indices) // self.batch_size

    def normalize_data(data, stats):
        min_stats = stats['min']
        max_stats = stats['max']
        
        # data will probably be (batch, horizon*32) and stats will be (32,), still want to normalize each state
        data = data.reshape(-1, 32)
        
        # nomalize to [0,1]
        
        # print(data.shape, stats['min'][:, None].shape, stats['max'][:, None].shape)
        
        ndata = (data - stats['min'][None, :]) / (stats['max'][None, :] - stats['min'][None, :])
        # normalize to [-1, 1]
        ndata = ndata * 2 - 1
        ndata = ndata.reshape(-1, 64)
        
        return ndata
    
    def normalize_data_actions(data, stats):
        # nomalize to [0,1]
        # min = stats['min'][:, None, None, None]
        
        ndata = (data - stats['min'][None, None, None, :]) / (stats['max'][None, None, None, :] - stats['min'][None, None, None, :])
        # normalize to [-1, 1]
        ndata = ndata * 2 - 1
        
        # print(ndata.shape, data.shape)
        # print(stats['min'][:, None, None, None].shape, stats['max'][:, None, None, None].shape)
        
        return ndata[0]

def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def get_data_stats_head(fin):
    data = fin['data']
    
    action_data = []
    state_data = []
    
    for demo in data:
        actions = np.array(data[demo]['actions'])
        action_data.append(actions)
        
        states = np.array(data[demo]['states'])
        state_data.append(states)

    action_data = np.concatenate(action_data, axis=0)
    state_data = np.concatenate(state_data, axis=0)

    info_action = get_data_stats(action_data)
    info_state = get_data_stats(state_data)
    
    return (info_action, info_state)

# Usage
# file_path = "demonstrations/1731007425_4282627/demo.hdf5"
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
#     # Your training loop or processing code here
#     print(batch["actions"].shape, batch["states"].shape)
#     time.sleep(60)
