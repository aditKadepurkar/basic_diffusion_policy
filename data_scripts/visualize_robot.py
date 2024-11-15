"""
This file is meant to be for visualizing the cleaned data
on the robot to see if we actually cleaned the data well
or if we ruined the data
"""

import sys
import os

sys.path.append(os.getcwd())

import h5py
import robosuite as suite
import argparse
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper


DATA_FILE = "demonstrations/data_norm.hdf5"
# DATA_FILE = "demonstrations/demo_norm.hdf5"

DEMO = "demo_2"


# model = Policy.model

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--directory",
    type=str,
    default=os.path.join(os.getcwd(), "demonstrations"),
)
parser.add_argument("--environment", type=str, default="Lift")
parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
parser.add_argument(
    "--config", type=str, default="single-arm-opposed", help="Specified environment configuration if necessary"
)
parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
parser.add_argument(
    "--controller", type=str, default="OSC_POSE", help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'"
)
parser.add_argument("--device", type=str, default="keyboard")
parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
args = parser.parse_args()

# Get controller config
controller_config = load_controller_config(default_controller=args.controller)

# Create argument configuration
config = {
    "env_name": args.environment,
    "robots": args.robots,
    "controller_configs": controller_config,
}

env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=128,
        camera_widths=128,
        camera_depths=False,
        reward_shaping=True,
        control_freq=20,
    )

env = GymWrapper(env)

observation = env.reset()

env.render()


with h5py.File(DATA_FILE, "r") as fin:
    for demo in fin['data'].keys():
        env.reset()
        data = fin['data'][demo]['actions']

    # print(data['states'].shape)
    # print(data['actions'].shape)

        for action in data:
            # print(action)
            action_setup = list(action[:3]) + list([0, 0, 0]) + [action[3]]
            # print(action_setup)
            # exit(0)
            env.step(action_setup)
            env.render()
    


