from collections import deque
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch as th

from rllte.common.base_storage import BaseStorage


class PrioritizedReplayStorage(BaseStorage):
    """Prioritized replay storage with proportional prioritization for off-policy algorithms.

    Args:
        observation_space (Space): The observation space of environment.
        action_space (Space): The action space of environment.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        storage_size (int): Max number of element in the buffer.
        batch_size (int): Batch size of samples.
        alpha (float): The alpha coefficient.
        beta (float): The beta coefficient.

    Returns:
        Prioritized replay storage.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        storage_size: int = 1000000,
        batch_size: int = 1024,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        super().__init__(observation_space, action_space, device)
        self.storage_size = storage_size
        self.batch_size = batch_size
        assert alpha > 0, "The prioritization value 'alpha' must be positive!"
        self.alpha = alpha
        self.beta = beta
        self.storage = deque(maxlen=storage_size)
        self.priorities = np.zeros((storage_size,), dtype=np.float32)
        self.position = 0

    def __len__(self):
        return len(self.storage)

    def annealing_beta(self, step: int) -> float:
        """Linearly increases beta from the initial value to 1 over global training steps.

        Args:
            step (int): The global training step.

        Returns:
            Beta value.
        """
        return min(1.0, self.beta + step * (1.0 - self.beta) / self.storage_size)

    def add(
        self,
        obs: Any,
        action: Any,
        reward: Any,
        terminated: Any,
        info: Any,
        next_obs: Any,
    ) -> None:
        """Add sampled transitions into storage.

        Args:
            obs (Any): Observations.
            action (Any): Actions.
            reward (Any): Rewards.
            terminated (Any): Terminateds.
            info (Any): Infos.
            next_obs (Any): Next observations.

        Returns:
            None.
        """
        transition = (obs, action, reward, terminated, next_obs)
        max_prio = self.priorities.max() if self.storage else 1.0
        self.priorities[self.position] = max_prio
        self.storage.append(transition)
        self.position = (self.position + 1) % self.storage_size

    def sample(self, step: int) -> Tuple[th.Tensor, ...]:
        """Sample from the storage.

        Args:
            step (int): Global training step.

        Returns:
            Batched samples.
        """
        if len(self.storage) == self.storage_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.position]

        probs = priorities**self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.storage), self.batch_size, p=probs)

        samples = [self.storage[i] for i in indices]
        weights = (len(self.storage) * probs[indices]) ** (-self.annealing_beta(step))
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        obs, actions, rewards, terminateds, next_obs = zip(*samples)
        obs = np.stack(obs)
        actions = np.stack(actions)
        rewards = np.expand_dims(np.stack(rewards), 1)
        terminateds = np.expand_dims(np.stack(terminateds), 1)
        next_obs = np.stack(next_obs)

        obs = th.as_tensor(obs, device=self.device).float()
        actions = th.as_tensor(actions, device=self.device).float()
        rewards = th.as_tensor(rewards, device=self.device).float()
        next_obs = th.as_tensor(next_obs, device=self.device).float()
        terminateds = th.as_tensor(terminateds, device=self.device).float()
        weights = th.as_tensor(weights, device=self.device).float()

        return indices, obs, actions, rewards, terminateds, next_obs, weights

    def update(self, metrics: Dict) -> None:
        """Update the priorities.

        Args:
            metrics (Dict): Training metrics from agent to udpate the priorities:
            indices (NdArray): The indices of current batch data.
            priorities (NdArray): The priorities of current batch data.

        Returns:
            None.
        """
        if "indices" in metrics and "priorities" in metrics:
            for i, priority in zip(metrics["indices"], metrics["priorities"]):
                self.priorities[i] = abs(priority)