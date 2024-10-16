import h5py
import jax.numpy as jnp
import jax

# should go to util file at some point
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    key = jax.random.PRNGKey(0)
    p = jax.random.permutation(key, len(a))
    return a[p], b[p]




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

                # print(f"States: {ret_states[0]}")
                # print(f"Actions: {ret_actions[0]}")

                ret_states1, ret_actions1 = unison_shuffled_copies(jnp.array(ret_states), jnp.array(ret_actions))

                # print(f"States: {ret_states1[0]}")
                # print(f"Actions: {ret_actions1[0]}")


                data_key['states'] = ret_states1
                data_key['actions'] = ret_actions1


                
                

                ret_data[key] = data_key

            

        f.close()

        return ret_data


# demo = { actions, states }
# 


