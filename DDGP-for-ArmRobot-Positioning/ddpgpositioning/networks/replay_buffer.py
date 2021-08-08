import torch
import numpy as np
from collections import namedtuple, deque


class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size):
        """Initialize a ReplayBuffer object.

        :param: buffer_size (int): maximum size of buffer
        """
        self.buffer = deque(maxlen=buffer_size)
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

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from memory.
        :param batch_size: number of samples in each batch
        """
        if self.__len__() == 0:
            return None
        indices = np.random.choice(
            self.__len__(), replace=False, size=min(batch_size, self.__len__())
        )
        return [self.buffer[idx] for idx in indices]


def convert_batch_to_tensor(experiences, device="cpu"):
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
        .float()
        .to(device)
    )

    return states, actions, rewards, next_states, dones
