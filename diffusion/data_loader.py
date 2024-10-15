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




            # print(data['demo_1'].keys())

            # print(data['demo_1']['states'])
            # print(data['demo_1']['actions'])
            
            
            
            # for i in range(self.index, self.index + count):
                # print("Keys: %s" % f.keys())
                
            #     if f.get(f"demo_{i}") is None:
            #         break
            #     actions = f[f"demo_{i}/actions"]
            # ret_data['observation'] = state





# loader = DataLoader("data/1728922451_627212/demo.hdf5")

# x = loader.load_data()

# # (38,) ie: the observation is 38 dimensional
# print(x[0]['states'].shape)

# # (4, 7) ie: 4 actions
# print(x[0]['actions'].shape)
