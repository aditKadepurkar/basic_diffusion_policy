"""
This file does a basic evaluation of the trained model.
"""

import jax
from jax import numpy as jnp
import robosuite as suite
from robosuite.wrappers import GymWrapper
import argparse
import os
from robosuite.controllers import load_controller_config
from diffusion.diffusion_policy import DiffusionPolicy


def eval_policy(Policy):
    """
    Evaluate the model on the given data_loader.
    """

    model = Policy.model

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

    obs = env.sim.get_state().flatten() # 1 step observation
    # extend the obs to 4 steps
    obs = jnp.broadcast_to(obs, (4, obs.shape[0]))

    # print(obs.shape)

    while not done:
        # Get the observation from the environment
        

        obs1 = env.sim.get_state().flatten()
        obs = jnp.concatenate([obs[1:], jnp.expand_dims(obs1, axis=0)], axis=0)

        # obs = jnp.array([obs])

        # print(obs.shape)

        # Pass the observation through the policy to get the predicted action
        action = DiffusionPolicy.inference(jnp.array([obs]), model, jax.random.PRNGKey(0))[0]

        # print(action.shape)

        # Take the action in the environment
        env.step(action[0])

        # Optionally render the environment
        env.render()

        obs1 = env.sim.get_state().flatten()
        obs = jnp.concatenate([obs[1:], jnp.expand_dims(obs1, axis=0)], axis=0)
        # obs = jnp.array([obs])

        env.step(action[1])

        env.render()



    print("Evaluation completed.")


# Example usage:
# model = ...  # Load or initialize your model
# data_loader = ...  # Define your data loader if necessary
# eval(Policy, model, data_loader)
