
import h5py

DATA_FILE = "demonstrations/demo_norm.hdf5"

with h5py.File(DATA_FILE, "r") as fin:
    data = fin['data']
    for demo in data:
        actions = data[demo]['actions']
        states = data[demo]['states']
        for i in range(actions.shape[0]):
            action = actions[i]
            state = states[i]
            print(action, state)


