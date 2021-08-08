from agent.ddpg import *
from agent.replay_buffer import *
import torch


class MADDPG(object):
    def __init__(self, **kwargs):
        """
        Initializes multiple agents using DDPG algorithm.
        Parameters
        ----------
        kwargs :
        """
        self.num_agents = kwargs.get("num_agents", 2)
        self.state_size = kwargs.get("state_size", 24)
        self.action_size = kwargs.get("action_size", 2)
        self.random_seed = int(kwargs.get("random_seed", 0))

        self.buffer_size = int(kwargs.get("buffer_size", 1e5))
        self.batch_size = kwargs.get("batch_size", 128)
        self.gamma = kwargs.get("gamma", 0.99)
        # Create n agents, where n = num_agents
        self.agents = [DDPG(**kwargs) for _ in range(self.num_agents)]
        self.alpha = kwargs.get("alpha", 0.7)

        # Create shared experience replay memory
        if kwargs.get("use_per", False):
            self.memory = PrioritizedReplayBuffer(
                self.buffer_size, self.alpha, self.random_seed
            )
        else:
            self.memory = ReplayBuffer(self.buffer_size, self.random_seed)

    def act(self, states, add_noise=True):
        """
        Perform action for multiple agents. Uses single agent act(), but now MARL
        Parameters
        ----------
        states :
        add_noise :

        Returns
        -------

        """
        actions = []
        for state, agent in zip(states, self.agents):
            action = agent.act(state, add_noise)  # get action from a single agent
            actions.append(action)
        return actions

    def reset(self):
        """
        Reset the noise level of multiple agents

        Returns
        -------

        """

        for agent in self.agents:
            agent.reset()

    def step(self, states, actions, rewards, next_states, dones, beta):
        """
        Saves an experience in the replay memory to learn from using random sampling.
        Parameters
        ----------
        states :
        actions :
        rewards :
        next_states :
        dones :

        Returns
        -------

        """

        # Save trajectories to Replay buffer
        for i in range(self.num_agents):
            self.memory.add(
                (states[i], actions[i], rewards[i], next_states[i], dones[i])
            )

        # check if enough samples in buffer. if so, learn from experiences, otherwise, keep collecting samples.
        if len(self.memory) > self.batch_size:
            for _ in range(self.num_agents):
                self.learn(beta)

    def learn(self, beta):
        """
        Learn from an agents experiences. performs batch learning for multiple agents simultaneously
        Parameters
        ----------
        beta :

        Returns
        -------

        """
        for agent in self.agents:
            experiences, batch_indices, batch_weights = self.memory.sample(
                    self.batch_size, beta
            )
            batch_weights_updated = agent.learn(experiences, batch_weights, self.gamma)
            self.memory.update(batch_indices, batch_weights_updated)
