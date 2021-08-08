import numpy as np
from networks.actor_critic import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agent.replay_buffer import convert_batch_to_tensor


class DDPG(object):
    """Main DDPG agent that extracts experiences and learns from them"""

    def __init__(self, **kwargs):
        """
        Initializes Agent object.

        Parameters
        ----------
        kwargs :
        """
        self.lr_critic = kwargs.get("rl_critic", 2e-4)
        self.rl_actor = kwargs.get("lr_actor", 2e-4)
        self.weight_decay = kwargs.get("weight_decay", 0.0)
        self.tau = kwargs.get("tau", 1e-2)

        self.state_size = kwargs.get("state_size", 24)
        self.action_size = kwargs.get("action_size", 2)
        self.random_seed = kwargs.get("random_seed", 0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Actor network
        self.actor_local = Actor(
            self.state_size, self.action_size, self.random_seed
        ).to(self.device)
        self.actor_target = Actor(
            self.state_size, self.action_size, self.random_seed
        ).to(self.device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=self.rl_actor
        )

        # Critic network
        self.critic_local = Critic(
            self.state_size, self.action_size, self.random_seed
        ).to(self.device)
        self.critic_target = Critic(
            self.state_size, self.action_size, self.random_seed
        ).to(self.device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=self.lr_critic,
            weight_decay=self.weight_decay,
        )

        # Perform hard copy
        self.soft_update(self.actor_target, self.actor_local, 1.0)
        self.soft_update(self.critic_target, self.critic_local, 1.0)

        # Noise proccess
        self.noise = OUNoise(
            self.action_size, self.random_seed
        )  # define Ornstein-Uhlenbeck process

    def reset(self):
        """
        Resets the noise process to mean
        Returns
        -------

        """

        self.noise.reset()

    def act(self, state, add_noise=True):
        """
        Returns a deterministic action given current state.
        Parameters
        ----------
        state :
        add_noise :

        Returns
        -------

        """

        state = (
            torch.from_numpy(state).float().to(self.device)
        )  # typecast to torch.Tensor
        self.actor_local.eval()  # set in evaluation mode
        with torch.no_grad():  # reset gradients
            action = (
                self.actor_local(state).cpu().data.numpy()
            )  # deterministic action based on Actor's forward pass.
        self.actor_local.train()  # set training mode

        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def learn(self, experiences, batch_weights, gamma):
        """
         Learn from a set of experiences picked up from a random sampling of even frequency (not prioritized)
        of experiences when buffer_size = MINI_BATCH.
        Updates policy and value parameters accordingly
        Parameters
        ----------
        experiences :
        batch_weights :
        gamma :

        Returns
        -------

        """

        # Extrapolate experience into (state, action, reward, next_state, done) tuples
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            batch_weights,
        ) = convert_batch_to_tensor(experiences, batch_weights, self.device)

        # Update Critic network
        actions_next = self.actor_target(
            next_states
        )  # Get predicted next-state actions and Q values from target models
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (
            gamma * Q_targets_next * (1 - dones)
        )  # r + γ * Q-values(a,s)

        # Compute critic loss using MSE
        Q_expected = self.critic_local(states, actions)
        batch_weights_updated = (batch_weights * (Q_expected - Q_targets) ** 2) + 1e-6
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)  # clip gradients
        self.critic_optimizer.step()

        # Update Actor Network

        # Compute actor loss
        actions_pred = self.actor_local(states)  # gets mu(s)
        actor_loss = -self.critic_local(states, actions_pred).mean()  # gets V(s,a)
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

        return batch_weights_updated.squeeze(1).data.cpu().numpy()

    def soft_update(self, local_model, target_model, tau):
        """
        "Soft update model parameters. Copies model τ every experience.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Parameters
        ----------
        local_model :
        target_model :
        tau :

        Returns
        -------

        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
