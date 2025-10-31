
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from collections import deque
import torch # Import torch

from config import Config, SEED # Import SEED
from agent import DuelingDDQNAgent


def train(cfg: Config):
    """Trains the Dueling DDQN agent."""
    # Add a print statement here to confirm the loaded total_steps inside the function
    print(f"train function received total_steps: {cfg.total_steps}")

    # Create environment
    env = gym.make(cfg.env_id)
    obs, info = env.reset(seed=SEED) # Use SEED directly
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Create agent
    agent = DuelingDDQNAgent(state_dim, n_actions, cfg, seed=SEED) # Use SEED directly

    # Training loop
    episode_returns = [] # List to store total return for each episode
    returns_window = deque(maxlen=100) # Store last 100 returns for average
    losses = deque(maxlen=100) # Store last 100 losses for average

    total_reward_episode = 0 # Accumulate reward within an episode

    pbar = tqdm(range(cfg.total_steps))
    for step in pbar:
        agent.global_step = step

        # Select action
        action = agent.act(obs)

        # Take step in environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Accumulate reward for the current episode
        total_reward_episode += reward

        # Clip reward if configured
        if cfg.reward_clip is not None:
            reward = np.clip(reward, -cfg.reward_clip, cfg.reward_clip)


        # Store transition in replay buffer
        agent.rb.add(obs, action, reward, next_obs, terminated or truncated)

        # Update agent (if enough data in replay buffer)
        loss = agent.update()
        if loss is not None:
            losses.append(loss)

        # Update epsilon
        agent.update_epsilon()

        # Reset environment if episode finished
        if terminated or truncated:
            episode_returns.append(total_reward_episode) # Store total reward for the finished episode
            returns_window.append(total_reward_episode) # Add to the window for average calculation
            total_reward_episode = 0 # Reset episode reward
            obs, info = env.reset()
        else:
            obs = next_obs

        # Log progress
        if step % 1000 == 0:
            avg_return = np.mean(list(returns_window)) if returns_window else 0 # Calculate average from window
            avg_loss = np.mean(losses) if losses else 0
            pbar.set_description(f"Step: {step}, Avg Return: {avg_return:.2f}, Avg Loss: {avg_loss:.4f}, Epsilon: {agent.eps:.3f}")

    env.close()
    return agent, episode_returns, losses # Return episode_returns
