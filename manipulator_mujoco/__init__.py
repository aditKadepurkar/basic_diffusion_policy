# Import the registration function from Gymnasium
from gymnasium.envs.registration import register

register(
    id="manipulator_mujoco/SimplifiedEnv-v0",
    entry_point="manipulator_mujoco.envs:SimplifiedEnv",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)

register(
    id="manipulator_mujoco/UR5eEnv-v0",
    entry_point="manipulator_mujoco.envs:UR5eEnv",
    # Optionally, you can set a maximum number of steps per episode
    # max_episode_steps=300,
    # TODO: Uncomment the above line if you want to set a maximum episode step limit
)
