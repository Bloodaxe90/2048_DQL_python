import random
from collections import deque


class ReplayBuffer(deque):

    def __init__(self, capacity: int):
        super().__init__(maxlen=capacity)

    def push(self, transition):
        super().append(transition)

    def full(self) -> bool:
        return len(self) >= self.maxlen

    def sample(self, batch_size: int):
        return random.sample(self, batch_size)