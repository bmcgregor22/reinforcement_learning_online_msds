
from collections import deque, namedtuple
import random
import numpy as np
import torch

# Define a namedtuple to represent a single transition in the environment
# A transition includes the state, the action taken, the reward received,
# the next state, and whether the episode ended (done flag).
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
  """
  A simple replay buffer to store and sample experiences for training.
  It uses a deque (double-ended queue) with a fixed maximum capacity.
  """
  def __init__(self, capacity: int, seed: int =42):
    '''
    Initializes the ReplayBuffer.

    Args:
        capacity: The maximum number of transitions to store in the buffer.
        seed: A random seed for reproducibility.
    '''
    # Create a deque with a maximum length. When the deque is full,
    # adding new elements automatically removes elements from the opposite end (FIFO).
    self.buffer = deque(maxlen=capacity)
    # Initialize a random number generator for sampling
    self.rng = random.Random(seed)

  def __len__(self):
    """Returns the current number of transitions stored in the buffer."""
    return len(self.buffer)

  def add(self, state, action, reward, next_state, done):
    """
    Adds a new transition to the replay buffer.

    Args:
        state: The current state observation.
        action: The action taken in the state.
        reward: The reward received after taking the action.
        next_state: The observation of the state after taking the action.
        done: A boolean flag indicating if the episode terminated or was truncated.
    """
    # Append the new transition (as a Transition namedtuple) to the right side of the deque.
    self.buffer.append(Transition(state, action, reward, next_state, done))

  def sample(self, batch_size: int, device: torch.device):
      """
      Randomly samples a minibatch of transitions from the buffer and
      returns them as PyTorch tensors, ready for training.

      Args:
          batch_size: The number of transitions to sample.
          device: The PyTorch device (e.g., 'cuda' or 'cpu') to place the tensors on.

      Returns:
          A tuple of PyTorch tensors: (states, actions, rewards, next_states, dones).
      """
      # Randomly sample 'batch_size' number of transitions from the buffer.
      # By sampling randomly from the replay buffer, you break correlations
      # between consecutive experiences, which is crucial for the stability and
      # effectiveness of off-policy reinforcement learning algorithms like DQN.
      batch = self.rng.sample(self.buffer, batch_size)

      # Convert the sampled batch of transitions into separate PyTorch tensors.
      # This step prepares the data in the required format for PyTorch neural networks.

      # Stack individual state NumPy arrays into a single batch NumPy array,
      # then convert to a PyTorch tensor of type float32 and move to the specified device.
      state  = torch.tensor(np.stack([b.state  for b in batch]), dtype=torch.float32, device=device)

      # Convert action list to a PyTorch tensor of type long (for discrete actions),
      # move to device, and add an extra dimension (unsqueeze(1)) for consistency with gathering operations.
      action  = torch.tensor([b.action   for b in batch], dtype=torch.long,   device=device).unsqueeze(1) # Actions are long tensors

      # Convert reward list to a PyTorch tensor of type float32,
      # move to device, and add an extra dimension (unsqueeze(1)) for consistency.
      reward  = torch.tensor([b.reward   for b in batch], dtype=torch.float32, device=device).unsqueeze(1) # Rewards are float tensors

      # Stack individual next_state NumPy arrays into a batch NumPy array,
      # then convert to a PyTorch tensor of type float32 and move to the specified device.
      next_state = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32, device=device)

      # Convert done list (boolean flags) to a PyTorch tensor of type float32 (0.0 or 1.0),
      # move to device, and add an extra dimension (unsqueeze(1)). Used for masking terminal states in target calculation.
      done  = torch.tensor([b.done for b in batch], dtype=torch.float32, device=device).unsqueeze(1)

      return state, action, reward, next_state, done
