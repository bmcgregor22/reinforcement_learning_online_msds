
from dataclasses import dataclass

SEED = 42 # Random seed for reproducibility

@dataclass
class Config:
    # Environment Parameters
    env_id: str = "LunarLander-v3" # Environment ID

    # Training Parameters
    total_steps: int = 200_000 # Total number of steps to train for
    start_learning: int = 10_000 # Number of steps before training starts
    buffer_size: int = 200_000 # Size of the replay buffer
    batch_size: int = 128 # Batch size for training
    gamma: float = 0.99 # Discount factor
    lr: float = 5e-4 # Learning rate for the optimizer 
    train_freq: int = 1 # Training frequency (train every N steps)
    target_update_freq: int = 1_000  # Frequency to update the target network (hard copy)
    tau: float = 0.005                 # Interpolation factor for soft target updates (1.0 for hard copy)
    grad_clip: float = 10.0 # Gradient clipping value
    reward_clip: float | None = 1.0  # Clip rewards to [-1, 1] if not None

    # Exploration Parameters
    eps_start: float = 1.0 # Starting value of epsilon for epsilon-greedy exploration
    eps_end: float = 0.05 # Ending value of epsilon for epsilon-greedy exploration
    eps_decay_steps: int = 200_000 # Number of steps over which epsilon decays
