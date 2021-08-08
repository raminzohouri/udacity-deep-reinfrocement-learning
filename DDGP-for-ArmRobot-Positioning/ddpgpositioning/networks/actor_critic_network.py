import torch
import torch.nn as nn
import numpy as np
import random
import copy


def reset_parameters(layers, w_init=3e-3):
    """

    Parameters
    ----------
    layers :
    w_init :

    Returns
    -------

    """
    for i, l in enumerate(layers):
        if type(l) == torch.nn.Linear:
            l.weight.data.uniform_(-w_init, w_init)


def build_model_layers(
    dims, use_batchnorm=False, hidden_activation=nn.ReLU(), output_activation=None
):
    """

    Parameters
    ----------
    dims :
    use_batchnorm :
    hidden_activation :
    output_activation :

    Returns
    -------

    """
    # Create the hidden layers
    model_layers = nn.ModuleList()
    for i, ds in enumerate(zip(dims[:-1], dims[1:])):
        dim_in, dim_out = ds
        model_layers.append(nn.Linear(dim_in, dim_out))
        if use_batchnorm:
            model_layers.append(nn.BatchNorm1d(dim_out))
        if i < len(dims) - 2:
            model_layers.append(hidden_activation)
    if output_activation is not None:
        model_layers.append(output_activation)
    return model_layers


def get_activation_function(activation_name=None):
    """

    Parameters
    ----------
    activation_name :

    Returns
    -------

    """
    activation_fun = None
    if activation_name is not None:
        if activation_name == "ReLU":
            activation_fun = nn.ReLU()
        elif activation_name == "Tanh":
            activation_name = nn.Tanh()
        elif activation_name == "Sigmoid":
            activation_fun = nn.Sigmoid()
    return activation_fun


class ActorNetwork(nn.Module):
    """Actor Network (Policy) Model."""

    def __init__(self, **kwargs):
        """
        Initialize parameters and build model.
        """
        super(ActorNetwork, self).__init__()
        self.state_size = kwargs.get("state_size", 33)
        self.action_size = kwargs.get("action_size", 4)
        self.seed = kwargs.get(
            "seed",
        )
        self.hidden_layers = kwargs.get("actor_hidden_layers", [400, 300])
        self.hidden_activation = get_activation_function(
            kwargs.get("actor_hidden_activation", None)
        )
        self.output_activation = get_activation_function(
            kwargs.get("actor_output_activation", None)
        )
        self.use_batchnorm = kwargs.get("actor_use_batchnorm", False)
        self.w_init = kwargs.get("w_init", 3e-3)
        self.seed = kwargs.get("seed", torch.manual_seed(47))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dims = [self.state_size] + self.hidden_layers + [self.action_size]
        self.actor_layers = build_model_layers(
            self.dims,
            self.use_batchnorm,
            self.hidden_activation,
            self.output_activation,
        )
        reset_parameters(self.actor_layers, self.w_init)

        self.to(self.device)
        print("Actor Network :", self.actor_layers)

    def forward(self, x):
        for i, l in enumerate(self.actor_layers):
            x = l(x)
        return x


class CriticNetwork(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, **kwargs):
        """Initialize parameters and build model."""
        super(CriticNetwork, self).__init__()
        self.state_size = kwargs.get("state_size", 33)
        self.action_size = kwargs.get("action_size", 4)
        self.seed = kwargs.get(
            "seed",
        )
        self.state_hidden_layers = kwargs.get("critic_state_hidden_layers", [400])
        self.action_hidden_layers = kwargs.get(
            "critic_action_hidden_layers", [400, 300]
        )
        self.hidden_activation = get_activation_function(
            kwargs.get("critic_hidden_activation", None)
        )
        self.output_activation = get_activation_function(
            kwargs.get("critic_output_activation", None)
        )
        self.use_batchnorm = kwargs.get("critic_use_batchnorm", False)
        self.w_init = kwargs.get("w_init", 3e-3)
        self.seed = kwargs.get("seed", torch.manual_seed(47))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert (
            self.state_hidden_layers[-1] == self.action_hidden_layers[0]
        ), "fanout of last hidden layer of state features should be equal to fanin of first action-state hidden layer "

        self.state_dims = [self.state_size] + self.state_hidden_layers
        self.action_dims = (
            [self.action_hidden_layers[0] + self.action_size]
            + self.action_hidden_layers[1:]
            + [1]
        )

        self.critic_state_layers = build_model_layers(
            self.state_dims, self.use_batchnorm, self.hidden_activation, nn.ReLU()
        )
        self.critic_action_layers = build_model_layers(
            self.action_dims, False, self.hidden_activation, self.output_activation
        )

        reset_parameters(self.critic_state_layers, self.w_init)
        reset_parameters(self.critic_action_layers, self.w_init)
        self.to(self.device)

        print(
            "Critic network built:", self.critic_state_layers, self.critic_action_layers
        )

    def forward(self, state, action):
        for i, l in enumerate(self.critic_state_layers):
            state = l(state)

        x = torch.cat((state, action), dim=1)
        for i, l in enumerate(self.critic_action_layers):
            x = l(x)
        return x


class OUActionNoise(object):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """
        Initialize parameters and noise process
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
