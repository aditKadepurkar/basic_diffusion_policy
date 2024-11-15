
import h5py
import numpy as np

DATA_FILE = "demonstrations/1731007425_4282627/demo.hdf5"

with h5py.File(DATA_FILE, "r") as fin:
    data = fin['data']
    for demo in data:
        actions = data[demo]['actions']
        states = data[demo]['states']

        print(actions.shape, states.shape)

        prev_action = actions[0]
        for i in range(1, actions.shape[0]):
            action = actions[i]
            state = states[i]

            print(action)

            # if np.linalg.norm(action[:len(prev_action)-1]) < 0.1 and prev_action[-1] == action[-1]:
            #     print(prev_action)
            #     print(action)
            #     print("no action")
            
            prev_action = action

            # print(len(action), len(state))


