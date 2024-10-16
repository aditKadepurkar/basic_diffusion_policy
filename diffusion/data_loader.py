import h5py







class DataLoader():
    def __init__(self, filename):
        self.filename = filename
        self.index = 0

    def load_data(self, count=4):
        ret_data = {}
        with h5py.File(self.filename, "r") as f:
            data = f['data']

            for key in data.keys():
                # extract the states and actions
                actions = data[key]['actions']
                states = data[key]['states']

                data_key = {}

                ret_states = []
                ret_actions = []

                for i in range(0, len(states)-4, 4):
                    ret_states.append(states[i])
                    ret_actions.append(actions[i:i+4])

                data_key['states'] = ret_states
                data_key['actions'] = ret_actions
                

                ret_data[key] = data_key


            return ret_data


# demo = { actions, states }
# 


