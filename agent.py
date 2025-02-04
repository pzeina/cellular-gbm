import gymnasium as gym
import tissue_env as tissue_env

env = gym.make("tissue_env/TerrainWorld-v0", render_mode="human")
observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated
        
    # Render the environment
    env.render()

env.reset()