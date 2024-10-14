import gymnasium as gym
import manipulator_mujoco  # Ensure this package is correctly installed and configured


# Create the environment again with rendering mode for visualization
env = gym.make('manipulator_mujoco/UR5eEnv-v0', render_mode='human')

# Reset the environment with a specific seed for reproducibility
observation, info = env.reset(seed=42)


# Run simulation to visualize the learned policy
while True:

    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    
    print(f"Reward: {reward}")

    if terminated or truncated:
        observation, info = env.reset()
        print("Episode over")
        break

# Close the environment when the simulation is done
env.close()
