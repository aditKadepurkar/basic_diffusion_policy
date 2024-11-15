"""
This file does a basic evaluation of the trained model.
"""

import torch
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
import argparse
import os
from robosuite.controllers import load_controller_config
from diffusion.diffusion_policy import DiffusionPolicy


def eval_policy(model, noise_scheduler, stats):
    """
    Evaluate the model on the given data_loader.
    """

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

    # Initialize the environment
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
    env = GymWrapper(env)  # Wrap the environment as a Gym environment for easier action and observation handling.

    # Reset environment to get the initial observation
    observation = env.reset()

    env.render()
    done = False

    obs = env.sim.get_state().flatten()  # 1 step observation

    obs = normalize_data(obs, stats[1])

    # extend the obs to 4 steps
    obs = np.tile(obs, 2)  # (128,) or something

    # try forcing the first few observations to see if it follows the expert
    # that would tell us if we just need more data, or something else.

    obs = torch.tensor(obs).float().to("cuda:0")
    # print(obs.shape)

    while not done:
        # Get the observation from the environment
        obs1 = env.sim.get_state().flatten()

        obs1 = torch.tensor(normalize_data(obs1, stats[1])).float().to("cuda:0")

        # each obs is 128/4 = 32
        obs = obs[:, 32:]

        # print(obs.shape, obs1.shape)

        obs = torch.concatenate([obs, obs1], axis=-1).float().to("cuda:0")

        action = torch.randn((1, 8, 4)).to("cuda:0")

        T = 50

        # TODO EVALUATE WHICH METHOD WORKS BETTER

        # METHOD 1!
        # for t in range(T, -1, -1):
        #     # print(action.shape, np.array([t]).shape, obs.shape)
        #     noise_pred = model(action, np.array([t]), obs)
        #     # print(noise_pred.shape)
        #     action = noise_scheduler.step(noise_pred, t, action)
        #     # print(action.shape)

        noise_scheduler.set_timesteps(50)

        with torch.no_grad():
            for k in noise_scheduler.timesteps:
                # predict noise
                # print(action.dtype, k.dtype, obs.dtype)
                k_val = torch.tensor([k]).to("cuda:0")
                # action = action[0]
                # action = torch.stack([action, action], dim=0)
                # obs = torch.stack([obs[0], obs[0]], dim=0)
                # print(action.shape, k_val.shape, obs.shape)

                noise_pred = model["noise_pred_net"](
                    sample=action,
                    timestep=k_val,
                    global_cond=obs
                )

                # inverse diffusion step (remove noise)
                action = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=action
                ).prev_sample

        # METHOD 2!
        # print(action.shape, T.shape, obs.shape)
        # action = model(action, T, obs)
        
        # print(action.shape)
        action = action[0][0]
        
        action = [action[0].item(), action[1].item(), action[2].item(), action[3].item()]
        
        # print(action)
        action = np.array(action)
        
        # action = unnormalize_data(action, stats[0])

        action = [action[0], action[1], action[2], 0, 0, 0, action[3]]
        # print(action)

        # Take the action in the environment
        env_obs, reward, finished, truncated, info = env.step(np.array(action))

        # print(env_obs.keys())
        
        # print("exiting, delete this line")
        # exit(0)

        env.render()

    print("Evaluation completed.")

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

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
    ndata = ndata.reshape(-1, 32)
    
    return ndata

# Example usage:
# model = ...  # Load or initialize your model
# data_loader = ...  # Define your data loader if necessary
# eval(Policy, model, data_loader)
