import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import math
import numpy as np
from collections import namedtuple, deque


#######
class QNetworkWithBatchNorm(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetworkWithBatchNorm, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc_features = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.BatchNorm1d(fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.BatchNorm1d(fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, fc2_units),
            nn.BatchNorm1d(fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, fc2_units),
            nn.BatchNorm1d(fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size),
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.fc_features(state)


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc_features = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size),
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.fc_features(state)


class DuelingNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DuelingNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc_features = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, fc2_units),
            nn.ReLU(),
        )
        self.fc_adv = nn.Sequential(
            nn.Linear(fc2_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size),
        )
        self.fc_val = nn.Sequential(
            nn.Linear(fc2_units, fc2_units), nn.ReLU(), nn.Linear(fc2_units, 1)
        )

    def forward(self, x):
        x = self.fc_features(x)
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + adv - adv.mean()
