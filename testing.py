import robosuite as suite
from robosuite.controllers import load_controller_config
import numpy as np

# Create a Robosuite environment with a UR5e robot for object reorientation
def create_reorientation_env():
    controller_config = load_controller_config(default_controller="OSC_POSE")

    env = suite.make(
        "Lift",
        robots="UR5e",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        controller_configs=controller_config,
        use_object_obs=True,
        reward_shaping=True 
    )
    return env


def main():
    env = create_reorientation_env()

    obs = env.reset()

    for _ in range(500):
        action = np.random.randn(env.robots[0].dof)

        obs, reward, done, info = env.step(action)
        
        env.render()

        print(f"Observation: {obs}, Reward: {reward}")

        if done:
            break

    env.close()

if __name__ == "__main__":
    main()
