"""
Dataloader made for the new dataset that I have found.

The dataset has the benefit that the actions are much
more vibrant(ie: in our old dataset the actions were 
always the same, but in this dataset the actions are
better captured).

The task is also different


This dataset is credit to https://sites.google.com/view/berkeley-ur5/home
"""

import numpy as np
import numpy as jnp
import jax
# import jax.numpy as jnp
import cv2


class DataLoaderNew():
    def __init__(self, batch_size=32) -> None:
        self.DATA_PATH = "demonstrations/tiger.npy"
        self.data_count = 0
        self.sizes = []
        data = np.load(self.DATA_PATH, allow_pickle=True)
        for i in range(data.shape[0]):
            print(self.data_count)
            self.data_count += data[i]['action'].shape[0] - 9
            self.sizes.append(self.data_count)
        self._indices = np.arange(self.data_count)
        # self._indices = np.concatenate([self._indices, self._indices, self._indices])
        # shuffle
        np.random.shuffle(self._indices)
        self.batch_size = batch_size
        self._start = 0

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

    def _load_data(self, indices):
        # sort the indices
        indices = sorted(indices)
        
        states = []
        visual = []
        actions = []
        
        # print(indices)
        
        dataset = np.load(self.DATA_PATH, allow_pickle=True)
        
        for i in indices:
            idx = np.searchsorted(self.sizes, i, side='right')
            if i == self.sizes[idx]:
                data = dataset[idx] # this is a dictionary



                vis = data['image'][i] # (2, 480, 640, 3)
                vis = np.repeat(vis, 2, axis=0)
                
                resized_vis = np.array([cv2.resize(image, (48, 64)) for image in vis])
                # swap axes 1 and 3
                resized_vis = np.swapaxes(resized_vis, 1, 3)
                
                # print(data.keys())
                visual.append()
                states.append(data['robot_state'][0] * 2)
                actions.append(data['action'][0:8])

            else:
                if idx > 0:
                    i -= self.sizes[idx - 1]

                data = dataset[idx]

                # print(data.keys())

                vis = data['image'][i:i+2] # (2, 480, 640, 3)
                
                resized_vis = np.array([cv2.resize(image, (48, 64)) for image in vis])
                # swap axes 1 and 3
                resized_vis = np.swapaxes(resized_vis, 1, 3)
                # print(resized_vis.shape)

                # states.append(jnp.ravel(data['robot_state'][i:i+2]))

                visual.append(jnp.array(resized_vis))
                states.append(data['robot_state'][i:i+2])
                actions.append(data['action'][i+1:i+9])



        
        
        
        data = {"states": jnp.array(states, dtype=jnp.float32), 
                "actions": jnp.array(actions, dtype=jnp.float32), 
                "visual": jnp.array(visual, dtype=jnp.float32)}
            
            # print(data['states'].shape, data['actions'].shape)
        del states, actions, dataset

        return data # states.shape = (batch, obs_horizon * obs_dim), actions.shape = (batch, action_horizon, action_dim)

    def shuffle_data(self):
        """Shuffle the indices if needed."""
        np.random.shuffle(self._indices)

    def get_batch_count(self):
        return len(self._indices) // self.batch_size

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

if __name__ == "__main__":
    # testing
    dataloader = DataLoaderNew(batch_size=10)

    for data in dataloader:
        print(data['states'].shape, data['actions'].shape)


