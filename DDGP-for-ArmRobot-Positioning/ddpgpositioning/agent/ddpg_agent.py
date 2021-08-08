import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np
from networks.actor_critic_network import ActorNetwork, CriticNetwork, OUActionNoise
from networks.replay_buffer import (
    convert_batch_to_tensor,
    ReplayBuffer,
)


class DDPGAgent(object):
    """Interacts with and learns from the environment."""

    def __init__(self, **kwargs):
        """Initialize an Agent object. """
        self.gamma = kwargs.get("gamma", 0.99)
        self.tau = kwargs.get("tau", 0.001)
        self.batch_size = kwargs.get("batch_size", 32)
        self.lr_actor = kwargs.get("lr_actor", 0.001)
        self.lr_critic = kwargs.get("lr_critic", 0.001)
        self.clip_gradient_critic = kwargs.get("clip_gradient_critic", False)
        self.clip_gradient_norm = kwargs.get("clip_gradient_norm", 1.0)
        self.clip_gradient_actor = kwargs.get("clip_gradient_actor", False)
        self.noise_decay = kwargs.get("noise_decay", 0.999)
        self.update_every = kwargs.get("update_every", 2)
        self.t_step = 0
        self.action_size = kwargs.get("action_size", 4)
        self.seed = torch.manual_seed(kwargs.get("seed", 47))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.buffer_size = int(kwargs.get("buffer_size", 1e5))
        self.combined_reply_buffer = kwargs.get("combined_reply_buffer", False)

        print("DDPG Agent hyperparameters:\n\t\n {}".format(kwargs))
        # Actor Network
        self.actor_local = ActorNetwork(**kwargs)
        self.actor_target = ActorNetwork(**kwargs)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=self.lr_actor
        )

        # Critic Network
        self.critic_local = CriticNetwork(**kwargs)
        self.critic_target = CriticNetwork(**kwargs)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=self.lr_critic
        )

        # Initialize target networks weights with the local networks ones
        self.soft_update(self.actor_local, self.actor_target, 1.0)
        self.soft_update(self.critic_local, self.critic_target, 1.0)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # Noise process
        self.noise = OUActionNoise(self.action_size, kwargs.get("seed", 47))
        self.noise_decay = kwargs.get("noise_decay", 0.999)

    def act(self, state, noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).data.cpu().numpy()
        self.actor_local.train()

        if noise:
            # Add noise to the action in order to explore the environment
            action += self.noise_decay * self.noise.sample()
            # Decay the noise process along the time
            self.noise_decay *= self.noise_decay

        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay buffer, and use random sample from buffer to learn."""
        # Save experience
        self.t_step += 1
        self.replay_buffer.add((state, action, reward, next_state, done))
        # Learn, if enough samples are available in memory
        if len(self.replay_buffer) > self.batch_size:
            experiences = self.replay_buffer.sample(self.batch_size)
            if self.combined_reply_buffer:
                experiences.append(
                    self.replay_buffer.experience(
                        state, action, reward, next_state, done
                    )
                )
            if self.t_step % self.update_every == 0:
                self.learn(experiences)

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s') tuples
        """
        states, actions, rewards, next_states, dones = convert_batch_to_tensor(
            experiences, self.device
        )

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions from actor_target model
        actions_next = self.actor_target(next_states)
        # Get predicted next-state Q-Values from critic_target model
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next)
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.clip_gradient_critic:
            torch.nn.utils.clip_grad_norm_(
                self.critic_local.parameters(), self.clip_gradient_norm
            )
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.clip_gradient_actor:
            torch.nn.utils.clip_grad_norm_(
                self.critic_local.parameters(), self.clip_gradient_norm
            )
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)

    def soft_update(self, local_model, target_model, tau=None):
        """ soft update model parameters.   θ_target = τ*θ_local + (1 - τ)*θ_target """
        if not tau:
            tau = self.tau
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
