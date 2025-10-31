
import gymnasium as gym
import torch
import numpy as np

from config import Config
from agent import DuelingDDQNAgent

def run_episodes(agent: DuelingDDQNAgent, env_id: str, num_episodes: int = 10, render: bool = True):
    """Runs episodes with a trained agent."""
    # Set render_mode to 'human' for rendering, or None otherwise
    env = gym.make(env_id, render_mode="human" if render else None)

    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        while not terminated and not truncated:
            # Agent selects action (remove torch.no_grad() if running from here, or use agent.act if it handles device)
            # Assuming agent.act handles the device correctly
            action = agent.act(obs)
            # Step in the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            obs = next_obs

        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

    env.close()

