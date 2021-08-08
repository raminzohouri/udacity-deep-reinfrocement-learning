import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
import copy


def hidden_init(layer):
    """

    Parameters
    ----------
    layer :

    Returns
    -------

    """
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Estimates the policy deterministically using tanh activation for continuous action space"""

    def __init__(self, state_size=24, action_size=2, seed=0, fc1=128, fc2=128):
        """

        Parameters
        ----------
        state_size :
        action_size :
        seed :
        fc1 :
        fc2 :
        """

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Layer 1
        self.fc1 = nn.Linear(state_size, fc1)
        self.bn1 = nn.BatchNorm1d(fc1)
        # Layer 2
        self.fc2 = nn.Linear(fc1, fc2)
        self.bn2 = nn.BatchNorm1d(fc2)
        # Output layer
        self.fc3 = nn.Linear(fc2, action_size)  # µ(s|θ) {Deterministic policy}

        # Initialize Weights
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameters by setting a noise from distribution following from its respective hidden unit size.
        Format for (-fx,fx) followed from the original paper.
        Returns
        -------

        """

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Performs a single forward pass to map (state,action) to policy, pi.
        Parameters
        ----------
        state :

        Returns
        -------

        """

        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        x = state
        # Layer #1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Layer #2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Output
        x = self.fc3(x)
        mu = torch.tanh(x)
        return mu


class Critic(nn.Module):
    """Value approximator V(pi) as Q(s, a|θ)"""

    def __init__(self, state_size=24, action_size=2, seed=0, fc1=128, fc2=128):
        """

        Parameters
        ----------
        state_size :
        action_size :
        seed :
        fc1 :
        fc2 :
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Layer 1
        self.fc1 = nn.Linear(state_size, fc1)
        self.bn1 = nn.BatchNorm1d(fc1)
        # Layer 2
        self.fc2 = nn.Linear(fc1 + action_size, fc2)
        # Output layer
        self.fc3 = nn.Linear(fc2, 1)  # Q-value

        # Initialize Weights
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameters by setting a noise from distribution following from its respective hidden unit size.
        Format for (-fx,fx) followed from the original paper.
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Performs a single forward pass to map (state,action) to Q-value
        Parameters
        ----------
        state :
        action :

        Returns
        -------

        """
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        # Layer #1
        x = self.fc1(state)
        x = F.relu(x)
        x = self.bn1(x)
        # Layer #2
        x = torch.cat(
            (x, action), dim=1
        )  # Concatenate state with action. Note that the specific way of passing x_state into
        # layer #2.
        x = self.fc2(x)
        x = F.relu(x)
        # Output
        value = self.fc3(x)
        return value


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.1):
        """
        Initialize parameters and noise process.
        Parameters
        ----------
        size :
        seed :
        mu :
        theta :
        sigma :
        """

        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for i in range(len(x))]
        )
        self.state = x + dx
        return self.state
