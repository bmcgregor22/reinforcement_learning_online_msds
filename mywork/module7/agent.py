
# agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from network import DuelingQNetwork
from replay_buffer import ReplayBuffer

class DuelingDDQNAgent:
    """
    Implements a Dueling Double Deep Q-Network (Dueling DDQN) agent.
    """
    def __init__(self, state_dim: int, n_actions: int, cfg: Config, seed: int = 42):
        """
        Initializes the Dueling DDQN agent.

        Args:
            state_dim: Dimension of the state space.
            n_actions: Number of possible actions.
            cfg: Configuration object containing hyperparameters.
            seed: Random seed for reproducibility.
        """
        self.cfg = cfg
        self.n_actions = n_actions
        # Determine the device to use for tensors (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize the online and target Q-networks
        self.q = DuelingQNetwork(state_dim, n_actions).to(self.device)
        self.q_target = DuelingQNetwork(state_dim, n_actions).to(self.device)
        # Copy initial weights from the online network to the target network
        self.q_target.load_state_dict(self.q.state_dict())
        # Set the target network to evaluation mode (gradients are not computed)
        self.q_target.eval()

        # Initialize the optimizer for the online network
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        # Initialize the replay buffer
        self.rb = ReplayBuffer(cfg.buffer_size, seed=seed)

        # Exploration parameter (epsilon)
        self.eps = cfg.eps_start
        # Global step counter
        self.global_step = 0

    @torch.no_grad() # Decorator to disable gradient computation within this method
    def act(self, state_np: np.ndarray) -> int:
        """
        Selects an action using an epsilon-greedy policy on the online network.

        Args:
            state_np: The current state observation as a NumPy array.

        Returns:
            The selected action as an integer.
        """
        # Epsilon-greedy exploration: choose random action with probability epsilon
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        # Greedy action selection: choose action with the highest Q-value
        else:
            # Convert state to tensor and add a batch dimension
            s = torch.tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            # Get Q-values from the online network
            q = self.q(s)                   # [1, A]
            # Return the index of the action with the maximum Q-value
            return int(q.argmax(dim=1))     # greedy action

    def update(self):
        """
        Performs one learning step (update) of the agent's Q-network.
        Samples a batch from the replay buffer and updates the online network
        using the Double DQN and Dueling DQN concepts.
        """
        # Skip learning if not enough transitions are in the replay buffer
        if len(self.rb) < self.cfg.start_learning:
            return None
        # Skip learning if not at the specified training frequency
        if self.global_step % self.cfg.train_freq != 0:
            return None

        # Sample a batch of transitions from the replay buffer
        s, a, r, ns, d = self.rb.sample(self.cfg.batch_size, self.device)

        # Compute the target Q-values (TD targets)
        with torch.no_grad(): # No gradient computation needed for target calculation
            # Double DQN: Select the next action using the online network
            next_q_online = self.q(ns)                                # [B, A]
            next_actions = next_q_online.argmax(dim=1, keepdim=True)  # [B, 1]
            # Evaluate the selected next action using the target network
            next_q_target = self.q_target(ns).gather(1, next_actions) # [B, 1]

            # Calculate the TD target: R + gamma * Q_target(next_state, next_action) * (1 - done)
            # (1 - done) ensures that the target is just the reward for terminal states
            target = r + self.cfg.gamma * (1.0 - d) * next_q_target   # [B, 1]

        # Compute the current Q-values for the actions taken in the batch
        q_sa = self.q(s).gather(1, a)                                 # [B, 1]

        # Compute the loss (difference between current Q and target Q)
        # Using Huber loss (SmoothL1Loss) which is less sensitive to outliers than MSE
        loss = nn.SmoothL1Loss()(q_sa, target)

        # Perform a gradient descent step to update the online network
        self.opt.zero_grad(set_to_none=True) # Clear previous gradients
        loss.backward()                      # Compute gradients
        # Clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()                      # Update network weights

        # Update the target network
        if self.cfg.tau >= 1.0:
            # Hard update: copy weights from online to target network periodically
            if self.global_step % self.cfg.target_update_freq == 0:
                self.q_target.load_state_dict(self.q.state_dict())
        else:
            # Soft update: smoothly interpolate between online and target weights
            with torch.no_grad():
                for p, tp in zip(self.q.parameters(), self.q_target.parameters()):
                    tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

        # Return the loss value
        return float(loss.item())

    def update_epsilon(self):
        """
        Updates the epsilon value for epsilon-greedy exploration.
        Decays linearly from eps_start to eps_end over eps_decay_steps.
        """
        # Calculate the fraction of decay steps completed
        frac = min(1.0, self.global_step / self.cfg.eps_decay_steps)
        # Linearly interpolate epsilon
        self.eps = self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)
