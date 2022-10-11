import torch


class DoubleBuffer:
    """Double buffer for enabling async copies"""

    def __init__(self, shape,  device, requires_grad=False):
        self.device = device
        self.buffers = tuple(
            torch.zeros(shape, device=device, requires_grad=requires_grad)
            for _ in range(2)
        )

        if requires_grad:
            for buffer in self.buffers:
                buffer.grad = torch.zeros_like(buffer, device=device)

    def __getitem__(self, index):
        return self.buffers[index]


class CircularBuffer:
    def __init__(self, item_shape, length, device):
        self.device = device
        self.buffer = torch.empty((length, *item_shape), device=device)
        self.index = torch.zeros(1, device=device, dtype=torch.long)

    def get_current(self):
        out = self.buffer[self.index].squeeze(0)
        self.index.copy_(self.next_index)
        return out

    @property
    def prev_index(self):
        return (self.index - 1) % self.buffer.shape[0]

    @property
    def next_index(self):
        return (self.index + 1) % self.buffer.shape[0]

    def push_back(self, tensor):
        self.buffer[self.prev_index] = tensor.to(self.device)