from collections import deque

class History:
    def __init__(self, size):
        self.size = size
        self.items = deque([], self.size)

    def put(self, item):
        self.items.append(item)

    def get(self):
        if len(self.items) == 0:
            return []
        items = list(self.items)
        last_idx = len(items)
        history = [0 for _ in range(self.size)]
        history[0:len(items)] = items
        for i in range(len(items), self.size):
            history[i] = items[-1]

        return history
