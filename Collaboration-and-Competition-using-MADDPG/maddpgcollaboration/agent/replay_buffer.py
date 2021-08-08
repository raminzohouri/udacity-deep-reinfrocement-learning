import numpy as np
import random
from collections import deque, namedtuple
import torch


class PrioritizedReplayBuffer(object):
    """ Prioritized experience buffer implementation """

    def __init__(
        self,
        buffer_size,
        prob_alpha,
        random_seed,
    ):
        """
        :param: buffer_size (int): maximum size of buffer
        :param: prob_alpha (int): prioritization exponent
        :param random_seed:
        """
        self.prob_alpha = prob_alpha
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.random_seed = np.random.seed(random_seed)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def __len__(self):
        return len(self.buffer)

    def add(self, sample):
        """
        Add a new experience to memory.
         :param sample: a tuple of (state, action, reward, next_state, done)
        """
        self.buffer.append(self.experience(*sample))
        try:
            max_prio = np.max(self.priorities) if len(self.priorities) > 0 else 1.0
        except ValueError:
            raise
        self.priorities.append(max_prio)

    def sample(self, batch_size, beta=0.4):
        """
        Randomly sample a batch of experiences from memory.
        :param batch_size: number of samples in each batch
        """
        if self.__len__() == 0:
            return None

        probs = np.asarray(self.priorities) ** self.prob_alpha
        probs /= probs.sum()
        indices = np.random.choice(
            self.__len__(), replace=False, size=min(batch_size, self.__len__()), p=probs
        )
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return (
            samples,
            indices,
            np.array(weights, dtype=np.float32).reshape((len(weights), 1)),
        )

    def update(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, random_seed):
        """Initialize a ReplayBuffer object.

        :param: buffer_size (int): maximum size of buffer
        """
        self.buffer = deque(maxlen=buffer_size)
        self.random_seed = random.seed(random_seed)
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)

    def add(self, sample):
        """
        Add a new experience to memory.
           :param sample: a tuple of (state, action, reward, next_state, done)
        """
        self.buffer.append(self.experience(*sample))

    def sample(self, batch_size, beta=0.0):
        """
        Randomly sample a batch of experiences from memory.
        :param batch_size: number of samples in each batch
        """
        if self.__len__() == 0:
            return None
        indices = np.random.choice(
            self.__len__(), replace=False, size=min(batch_size, self.__len__())
        )
        return [self.buffer[idx] for idx in indices], indices, np.ones(len(indices))

    def update(self, indices, weights):
        pass


def convert_batch_to_tensor(experiences, batch_weights, device="cpu"):
    """
     Receieves batch of experience and convert it to torch tensor
    :param experiences: list oftuple of (s, a, r, s', done) tuples
    :param batch_weights: list priority weights per experiens, the value is None for regular buffer
    :param device: device type for torch tensor
    """
    states = (
        torch.from_numpy(np.vstack([e.state for e in experiences if e is not None]))
        .float()
        .to(device)
    )
    actions = (
        torch.from_numpy(np.vstack([e.action for e in experiences if e is not None]))
        .float()
        .to(device)
    )
    rewards = (
        torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None]))
        .float()
        .to(device)
    )
    next_states = (
        torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        )
        .float()
        .to(device)
    )
    dones = (
        torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        )
        .long()
        .to(device)
    )
    batch_weights = torch.tensor(batch_weights).to(device).float()
    return states, actions, rewards, next_states, dones, batch_weights
