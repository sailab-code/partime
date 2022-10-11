from typing import Tuple
import torch

from partime.buffers import DoubleBuffer

class Stage:

    module: torch.nn.Module

    proc_stream: torch.cuda.Stream
    receive_copy_stream: torch.cuda.Stream
    send_grad_stream: torch.cuda.Stream

    input_buffers: Tuple[DoubleBuffer, ...]
    gradients_buffers: torch.Tensor
    output_buffer: torch.Tensor

    optimizer: torch.optim.Optimizer

    def __init__(
        self,
        index: int,
        module: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer
    ):
        self.index = index
        self.module = module
        self.device = device
        self.optimizer = optimizer

        self.proc_stream = torch.cuda.Stream(device, priority=-1)
        self.receive_copy_stream = torch.cuda.Stream(device, priority=-1)
        self.send_grad_stream = torch.cuda.Stream(device, priority=-1)

        self.input_buffers: Tuple[DoubleBuffer, ...] = None
        self.gradients_buffers: Tuple[torch.Tensor, ...] = None

    def get_input_buffers(self, clock: int):
        return [
            sub_input_buffer[clock]
            for sub_input_buffer in self.input_buffers
        ]