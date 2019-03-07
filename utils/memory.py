import random


class Memory(object):
    def __init__(self, max_memory=100):
        self.max_memory = max_memory
        self.memory = list()

    def add(self, m):
        if len(self.memory) == self.max_memory:
            self.memory.pop(0)
        self.memory.append(m)

    def __len__(self):
        return len(self.memory)

    def get(self, batch_size):
        return random.sample(self.memory, batch_size)
