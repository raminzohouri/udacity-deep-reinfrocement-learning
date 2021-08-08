import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
from networks.q_network import QNetworkWithBatchNorm, QNetwork, DuelingNetwork
from networks.replay_buffer import (
    convert_batch_to_tensor,
    ReplayBuffer,
    PrioritizedReplayBuffer,
)


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, **kwargs):
        """
        initialize an Agent object.

        :param state_size (int): dimension of each state
        :param: action_size (int): dimension of each action
        :param: seed (int): random seed
        """
        # ---- general parameters --------#
        self.seed = kwargs.get("seed", 47)
        np.random.seed(self.seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.agent_description = kwargs.get("agent_description", "vanilla DQN")
        self.agent_name = kwargs.get("agent_name", "v_dqn")

        # ---- q network parameters ------#
        self.dueling_dqn = kwargs.get("dueling_dqn", False)
        self.with_batch_norm = kwargs.get("with_batch_norm", False)
        self.state_size = kwargs.get("state_size", 37)
        self.action_size = kwargs.get("action_size", 4)
        self.load_model = kwargs.get("load_model", False)

        # ---- replay buffer parameters --#
        self.use_per = kwargs.get("use_per", False)
        self.buffer_size = int(kwargs.get("buffer_size", 1e4))
        self.combined_reply_buffer = kwargs.get("combined_reply_buffer", False)
        self.alpha = kwargs.get("alpha", 0.7)

        # ---- DQN training parameters ---#
        self.gamma = kwargs.get("gamma", 0.99)
        self.learning_rate = kwargs.get("learning_rate", 0.00025)
        self.tau = kwargs.get("tau", 0.001)
        self.gradient_clip_norm = kwargs.get("gradient_clip_norm", 5.0)
        self.update_every = kwargs.get("update_every", 1)
        self.exploration_steps = kwargs.get("exploration_steps", 1000)
        self.target_network_update_freq = kwargs.get("target_network_update_freq", 4)
        self.batch_size = kwargs.get("batch_size", 32)
        self.use_soft_update = kwargs.get("use_soft_update", True)
        self.double_dqn = kwargs.get("double_dqn", False)

        # ---- build q network & replay memory -#

        # Replay memory
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(self.buffer_size, self.alpha)
        else:
            self.memory = ReplayBuffer(
                self.buffer_size,
            )

        if not self.dueling_dqn:
            if self.with_batch_norm:
                self.qnetwork_online = QNetworkWithBatchNorm(
                    self.state_size, self.action_size, self.seed
                ).to(self.device)
                self.qnetwork_target = QNetworkWithBatchNorm(
                    self.state_size, self.action_size, self.seed
                ).to(self.device)
            else:
                self.qnetwork_online = QNetwork(
                    self.state_size, self.action_size, self.seed
                ).to(self.device)
                self.qnetwork_target = QNetwork(
                    self.state_size, self.action_size, self.seed
                ).to(self.device)
        else:
            self.qnetwork_online = DuelingNetwork(
                self.state_size, self.action_size, self.seed
            ).to(self.device)
            self.qnetwork_target = DuelingNetwork(
                self.state_size, self.action_size, self.seed
            ).to(self.device)

        self.optimizer = optim.Adam(
            self.qnetwork_online.parameters(), lr=self.learning_rate
        )

        if self.load_model:
            self.qnetwork_online.load_state_dict(kwargs.get("model_state_dict"), None)
            self.qnetwork_target.load_state_dict(kwargs.get("model_state_dict"), None)
            self.optimizer.load_state_dict(kwargs.get("optimizer_state_dict"), None)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, beta):
        """
        apply one step of learning and add the new sample to the memory
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :param beta: important sampling exponent for calculating weights of prioritized samples
        """
        self.t_step += 1
        # Save experience in replay memory
        self.memory.add((state, action, reward, next_state, done))
        # Learn every UPDATE_EVERY time steps.
        if self.t_step > self.exploration_steps:
            # If enough samples are available in memory, get random subset and learn
            if self.use_per:
                experiences, batch_indices, batch_weights = self.memory.sample(
                    self.batch_size, beta
                )
            else:
                experiences = self.memory.sample(self.batch_size)
                if self.combined_reply_buffer:
                    experiences.append(
                        self.memory.experience(state, action, reward, next_state, done)
                    )
                batch_indices = None
                batch_weights = np.ones((len(experiences), 1))
            if self.t_step % self.update_every == 0:
                self.learn(experiences, batch_indices, batch_weights)

    def act(self, state, eps=0.1):
        """
        returns actions for given state as per current policy.
        :param: state (array_like): current state
        :param: eps (float): epsilon, for epsilon-greedy action selection
        """
        # Epsilon-greedy action selection
        if np.random.rand() > eps:
            state = torch.from_numpy(state).float().unsqueeze(0)
            self.qnetwork_online.eval()
            with torch.no_grad():
                action_values = self.qnetwork_online(state.to(self.device))
            self.qnetwork_online.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(self.action_size, replace=False)

    def calculate_loss(
        self,
        experiences,
        qnetwork_online,
        qnetwork_target,
        gamma,
        batch_weights,
        double_dqn=False,
        device="cpu",
    ):
        """
        caculated loss given online network and target network
        :param experiences: list oftuple of (s, a, r, s', done) tuples
        :param: qnetwork_online: behaviore policy netework for online interaction with environment
        :param: qnetwork_target: target policty netwrok calculating expected value
        :param: gamma: discount factor for Belman equation
        :param: batch_weights:  list priority weights per experiens, the value is None for regular buffer
        :param: double_dqn: if doulbe DQN algorithm for calcuting expected value
        :param device: device type for torch tensor
        """

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            batch_weights,
        ) = convert_batch_to_tensor(experiences, batch_weights, device)
        state_action_values = (
            qnetwork_online(states).gather(1, actions.squeeze(0)).squeeze(0)
        )
        with torch.no_grad():
            if double_dqn:
                next_state_values = (
                    qnetwork_target(next_states)
                    .gather(
                        1,
                        torch.argmax(qnetwork_online(next_states), dim=-1).unsqueeze(
                            -1
                        ),
                    )
                    .squeeze(-1)
                )
            else:
                next_state_values = qnetwork_target(next_states).max(1)[0]
        next_state_values[dones] = 0.0
        expected_state_action_values = (
            next_state_values.unsqueeze(1).detach() * gamma + rewards
        )
        losses = (
            batch_weights * (state_action_values - expected_state_action_values) ** 2
        )
        return losses.mean(), losses + 1e-6

    def learn(self, experiences, batch_indices, batch_weights):
        """
         update value parameters using given batch of experience tuples.

        :param: experiences (list): tuple of (s, a, r, s', done) tuples
        :param: gamma (float): discount factor
        """
        loss, sample_prios = self.calculate_loss(
            experiences,
            self.qnetwork_online,
            self.qnetwork_target,
            self.gamma,
            batch_weights,
            self.double_dqn,
            self.device,
        )
        if self.use_per:
            self.memory.update_priorities(
                batch_indices, sample_prios.squeeze(1).data.cpu().numpy()
            )

        # -------------------- apply gradient -----------------------#
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        nn.utils.clip_grad_norm_(
            self.qnetwork_online.parameters(), self.gradient_clip_norm
        )
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        if (self.t_step / self.update_every) % self.target_network_update_freq == 0:
            if self.use_soft_update:
                """
                soft update model parameters.   θ_target = τ*θ_local + (1 - τ)*θ_target
                """
                for target_param, online_param in zip(
                    self.qnetwork_target.parameters(), self.qnetwork_online.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * online_param.data
                        + (1.0 - self.tau) * target_param.data
                    )
            else:
                self.qnetwork_target.load_state_dict(self.qnetwork_online.state_dict())
