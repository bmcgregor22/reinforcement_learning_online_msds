
import torch
import torch.nn as nn
from typing import Tuple

class DuelingQNetwork(nn.Module): ##inherits from torch.nn.Module
    """
    Dueling Deep Q-Network (Dueling DQN) architecture.

    This network implements the dueling architecture by splitting the network
    into a shared feature extractor, a value stream (V), and an advantage
    stream (A). The Q-values are computed by combining V and A.
    """
    def __init__(self, state_dim: int, n_actions: int, hidden: Tuple[int, int] = (256, 256)):
        """
        Initializes the DuelingQNetwork.

        Args:
            state_dim: The dimension of the input state space.
            n_actions: The number of possible actions in the environment.
            hidden: A tuple specifying the number of units in the hidden layers
                    of the shared feature extractor.
        """
        super().__init__() #calls the parent class init to set up nn

        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
        )

        # Value stream (estimates V(s))
        self.value = nn.Sequential(
            nn.Linear(hidden[1], 128),
            nn.ReLU(),
            nn.Linear(128, 1), # Output a single value for V(s)
        )

        # Advantage stream (estimates A(s, a))
        self.advantage = nn.Sequential(
            nn.Linear(hidden[1], 128),
            nn.ReLU(),
            nn.Linear(128, n_actions), # Output a value for each action
        )

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x: The input tensor representing a batch of states.

        Returns:
            A tensor of Q-values for each action in the input states.
        """
        # Pass input through the shared feature extractor
        z = self.feature(x)

        # Get the state value from the value stream
        v = self.value(z)                        # [B, 1]

        # Get the advantage for each action from the advantage stream
        a = self.advantage(z)                    # [B, A]

        # Center the advantages for identifiability
        # Subtracting the mean of advantages doesn't change the relative order
        # of actions but helps in separating the roles of V and A.
        a = a - a.mean(dim=1, keepdim=True)      # center A for identifiability

        # Combine value and advantage to get Q-values
        # Q(s, a) = V(s) + A(s, a)
        return v + a                              # [B, A]
