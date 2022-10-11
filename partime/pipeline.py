from contextlib import contextmanager
from typing import Callable, List, Tuple, Union
import torch
from torch import Tensor, nn
from dataclasses import dataclass
from collections import OrderedDict
import copy
from inspect import signature

from partime.buffers import CircularBuffer, DoubleBuffer

from partime.stage import Stage


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        defaults = dict()
        super(DummyOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        pass
class Pipeline(nn.Module):

    @staticmethod
    def __split_module(
        module: nn.Sequential,
        balance: List[int],
        devices: List[torch.device],
        optimizer_initializer: Callable[[nn.Module], torch.optim.Optimizer]
    ) -> List[Stage]:

        stages: List[Stage] = []
        named_layers = list(module.named_children())

        # balance sum must be equal to len(modules) or the last value must be -1
        if sum(balance) != len(named_layers) and balance[-1] != -1:
            raise ValueError(f"Balance sum ({sum(balance)}) must be equal to len(modules) ({len(named_layers)}) or the last value must be -1")

        # devices must not be empty
        if len(devices) == 0:
            raise ValueError("Devices must not be empty")

        stages: List[Stage] = []

        # split named_layers into partitions, which sizes are given by balance

        n_partitions = len(balance)
        n_devices = len(devices)

        current_layer_idx = 0
        for i in range(n_partitions):
            balance_value = balance[i]
            if balance_value == -1:
                # take all remaining layers
                layers = named_layers[current_layer_idx:]
            else:
                # take balance_value layers
                layers = named_layers[current_layer_idx:current_layer_idx + balance_value]

            layers = [copy.deepcopy(layer) for layer in layers]

            # select device by placing nearby partitions in the same device
            dev_id = (i * n_devices) // n_partitions
            device = devices[dev_id]

            module = nn.Sequential(OrderedDict(layers)).to(device)

            optimizer = optimizer_initializer(module) if optimizer_initializer is not None else None
            # create partition with sequential module from layers
            stage = Stage(
                index=i,
                module=module,
                device=device,
                optimizer=optimizer,
            )

            # add partition to partitions
            stages.append(stage)

            # update current_layer_idx
            current_layer_idx += balance_value

        return stages

    def __init__(
        self,
        model: nn.Sequential,
        sample_input: torch.Tensor,
        balance: List[int] = [-1],
        devices: Union[List[torch.device], List[int]] = None,
        use_cuda_graph: bool = True,
        loss_function: Callable[[Tensor], Tensor] = None,
        sample_target: torch.Tensor = None,
        optimizer_settings: dict = None):
        # model must be a nn.Sequential model
        if not isinstance(model, nn.Sequential):
            raise ValueError("model must be a nn.Sequential model")

        # balance must be a list
        if not isinstance(balance, list):
            raise ValueError("balance must be a list")

        # check if all are ints
        if not all(isinstance(x, int) for x in balance):
            raise ValueError("balance must be a list of ints")

        # devices must be a list
        if not isinstance(devices, list):
            raise ValueError("devices must be a list")

        # check if all are torch.device or int
        if not all(isinstance(x, (torch.device, int, str)) for x in devices):
            raise ValueError("devices must be a list of torch.device or ints that indexes cuda devices or device ids1")

        # normalize to torch.device the devices that are ints
        devices = [torch.device(f"cuda:{x}") if isinstance(x, int) else x for x in devices]

        # normalize to torch.device the devices that are strings
        devices = [torch.device(x) if isinstance(x, str) else x for x in devices]

        # check if all devices are CPU or CUDA
        if not (all(x.type == "cpu" for x in devices) or all(x.type == "cuda" for x in devices)):
            raise ValueError("devices must be the same type (all cpu or all cuda)")

        # TODO: remove this check when CPU is supported
        # check if all devices are CUDA
        if any(x.type == "cpu" for x in devices):
            raise ValueError("CPU is not supported")

        self.devices = devices

        def optimizer_initializer(module: nn.Module) -> torch.optim.Optimizer:
            if optimizer_settings is None:
                return DummyOptimizer(module.parameters())
            else:
                optim_class = optimizer_settings[0]
                optim_kwargs = optimizer_settings[1]
                return optim_class(module.parameters(), **optim_kwargs)

        # split model into partitions
        self.stages = Pipeline.__split_module(model, balance, devices, optimizer_initializer)

        self.loss_fn = loss_function

        # check if loss_fn is supervised or not
        if loss_function is not None:
            self.is_loss_supervised = len(signature(loss_function).parameters) == 2
            print(f"is_loss_supervised: {self.is_loss_supervised}")

        self.use_cuda_graph = use_cuda_graph

        self.main_stream: torch.cuda.Stream = torch.cuda.Stream(self.devices[0], priority=-1)

        self.input_buffers: Tuple[torch.Tensor, ...] = None
        self.ouput_buffer: Tuple[torch.Tensor, ...] = None
        self.target_buffers: Tuple[CircularBuffer, ...] = None

        # TODO: write doc on this
        self.input_buffers_clock: int = 0
        self._is_clock_alternating = not self.use_cuda_graph

        self._allocate_buffers(sample_input, sample_target)
        self.frame_t = 0


        self.backward_enabled = self.loss_fn is not None

        self._graph: torch.cuda.CUDAGraph = None
        if self.use_cuda_graph:
            self._record_cuda_graph()

    @property
    def next_input_buffer_clock(self): # TODO: choose better name
        return (self.input_buffers_clock + 1) % 2 # if self._is_clock_alternating else self.input_buffers_clock

    @contextmanager
    def _pipeline_context(self):
        if not self.backward_enabled:
            with torch.no_grad():
                yield
        else:
            yield

    def _advance_clock(self):
        self.input_buffers_clock = self.next_input_buffer_clock

    def _zero_grad_weights(self, set_to_none=False):
        for stage in self.stages:
            with torch.cuda.stream(stage.proc_stream):
                stage.optimizer.zero_grad(set_to_none)

    def _allocate_buffers(self, sample_inputs: Tuple[torch.Tensor, ...], sample_targets: Tuple[torch.Tensor, ...] = None):

        if not isinstance(sample_inputs, tuple) and isinstance(sample_inputs, torch.Tensor):
            sample_inputs = (sample_inputs,)

        if sample_targets is not None and not isinstance(sample_targets, tuple) and isinstance(sample_targets, torch.Tensor):
            sample_targets = (sample_targets,)

        with torch.no_grad():
            self.input_buffers = tuple(
                torch.zeros_like(sub_input, device='cpu', pin_memory=True)
                for sub_input in sample_inputs
            )

            current_inputs = sample_inputs # set it as a tuple for consistency
            for stage_idx, stage in enumerate(self.stages):
                stage_requires_grad = stage_idx > 0
                # create input buffers for the stage

                stage.input_buffers = tuple(
                    DoubleBuffer(
                        sub_input.shape,
                        stage.device,
                        requires_grad=stage_requires_grad
                    )
                    for sub_input in current_inputs
                )

                current_inputs = tuple(sub_input.to(stage.device) for sub_input in current_inputs)

                if len(current_inputs) == 1:
                    outputs = stage.module(current_inputs[0])
                else:
                    outputs = stage.module(current_inputs)

                if not isinstance(outputs, tuple):
                    outputs = (outputs,)

                # create buffer from gradients from next stage
                if stage_idx < len(self.stages) - 1:
                    stage.gradients_buffers = tuple(
                        torch.zeros_like(
                            sub_output,
                            device=stage.device,
                            requires_grad=False
                        )
                        for sub_output in outputs
                    )

                current_inputs = outputs

            if not isinstance(outputs, tuple):
                outputs = (outputs,)

            self.output_buffers = tuple(
                torch.zeros_like(sub_output, device='cpu', pin_memory=True)
                for sub_output in outputs
            )

            if self.loss_fn is not None:
                if self.is_loss_supervised:
                    self.target_buffers = tuple(
                        CircularBuffer(
                            item_shape=sample_target.shape,
                            length=len(self.stages),
                            device=self.stages[-1].device,
                        )
                        for sample_target in sample_targets
                    )
                else:
                    self.target_buffers = tuple(
                        CircularBuffer(
                            item_shape=sub_output.shape,
                            length=len(self.stages),
                            device=sub_output.device,
                        )
                        for sub_output in outputs
                    )

            self.loss_buffer = torch.empty(1, device='cpu', pin_memory=True)

    def _record_cuda_graph(self):
        # cuda graph warmup
        for input_buffer in self.input_buffers:
            input_buffer.copy_(torch.rand_like(input_buffer))
        self.main_stream.wait_stream(torch.cuda.current_stream(self.devices[0]))
        with torch.cuda.stream(self.main_stream):
            for _ in range(0, 3):
                self._zero_grad_weights(set_to_none=True)
                self._step()
        torch.cuda.current_stream(self.devices[0]).wait_stream(self.main_stream)
        self.main_stream.synchronize()

        self._graph = torch.cuda.CUDAGraph()
        self._zero_grad_weights(set_to_none=True)
        with torch.cuda.graph(self._graph, stream=self.main_stream, pool=None):
            self._step()
        torch.cuda.current_stream(self.devices[0]).wait_stream(self.main_stream)
        self.main_stream.synchronize()

    def _step(self):
        with self._pipeline_context():
            # branch streams
            for stage_idx, stage in enumerate(self.stages):
                stage.proc_stream.wait_stream(self.main_stream)
                stage.receive_copy_stream.wait_stream(self.main_stream)
                if stage_idx > 0:
                    stage.send_grad_stream.wait_stream(self.main_stream)

            # receive gradients by the next stage (except for last partition)
            if self.backward_enabled:
                if not self.use_cuda_graph:
                    self._zero_grad_weights()
                for stage, next_stage in zip(self.stages[:-1], self.stages[1:]):
                    with torch.cuda.stream(next_stage.send_grad_stream), torch.cuda.stream(stage.receive_copy_stream):
                        with torch.no_grad():
                            next_stage_input_buffers = next_stage.get_input_buffers(self.next_input_buffer_clock)

                            for buffer_idx in range(len(next_stage_input_buffers)):
                                stage.gradients_buffers[buffer_idx].copy_(
                                    next_stage_input_buffers[buffer_idx].grad,
                                    non_blocking=True
                                )

                                next_stage_input_buffers[buffer_idx].grad.zero_()

            # forward and copy output
            outputs = [None] * len(self.stages)
            for stage_idx, stage in enumerate(self.stages):
                with torch.cuda.stream(stage.proc_stream):
                    input_buffers = stage.get_input_buffers(self.input_buffers_clock)
                    with torch.no_grad():
                        if stage_idx == 0:
                            for buffer_idx in range(len(input_buffers)):
                                input_buffers[buffer_idx].copy_(self.input_buffers[buffer_idx], non_blocking=True)
                        else:
                            if not self._is_clock_alternating:
                                stage_next_input_buffers = stage.get_input_buffers(self.next_input_buffer_clock)
                                for buffer_idx in range(len(input_buffers)):
                                    input_buffers[buffer_idx].copy_(stage_next_input_buffers[buffer_idx])

                    if len(input_buffers) == 1:
                        stage_output = stage.module(input_buffers[0])
                    else:
                        stage_output = stage.module(input_buffers)

                    if not isinstance(stage_output, tuple):
                        stage_output = (stage_output,)

                    outputs[stage_idx] = stage_output

                    if stage_idx < len(self.stages) - 1:
                        next_stage = self.stages[stage_idx + 1]
                        next_stage.receive_copy_stream.wait_stream(next_stage.send_grad_stream)
                        next_stage.receive_copy_stream.wait_stream(stage.proc_stream)
                        with torch.cuda.stream(next_stage.receive_copy_stream), torch.no_grad():
                            next_stage_input_buffers = next_stage.get_input_buffers(self.next_input_buffer_clock)

                            for buffer_idx in range(len(next_stage_input_buffers)):
                                next_stage_input_buffers[buffer_idx].copy_(outputs[stage_idx][buffer_idx], non_blocking=True)
                    else:
                        with torch.no_grad():
                            for buffer_idx in range(len(outputs[stage_idx])):
                                self.output_buffers[buffer_idx].copy_(outputs[stage_idx][buffer_idx], non_blocking=True)

            # loss computation and backward
            if self.backward_enabled:
                for stage_idx, stage in enumerate(self.stages[:-1]):
                    with torch.cuda.stream(stage.proc_stream):
                        stage_outputs = outputs[stage_idx]

                        torch.autograd.backward(
                                tensors=stage_outputs,
                                grad_tensors=stage.gradients_buffers
                        )

                        stage.optimizer.step()

                with torch.cuda.stream(self.stages[-1].proc_stream):
                    if self.is_loss_supervised:
                        out_loss = self.loss_fn(*outputs[-1], *[buffer.get_current().squeeze(0) for buffer in self.target_buffers])
                    else:
                        out_loss = self.loss_fn(*outputs[-1])
                    out_loss.backward()
                    self.stages[-1].optimizer.step()

                with torch.cuda.stream(self.stages[-1].receive_copy_stream):
                    self.loss_buffer.copy_(out_loss.detach(), non_blocking=True)

            # re-join streams
            for stage_idx, stage in enumerate(self.stages):
                self.main_stream.wait_stream(stage.proc_stream)
                self.main_stream.wait_stream(stage.receive_copy_stream)
                if stage_idx > 0:
                    self.main_stream.wait_stream(stage.send_grad_stream)

    def forward(self, input_tensors: Tuple[torch.Tensor, ...] = None, target_tensors: Tuple[torch.Tensor, ...] = None):
        if input_tensors is not None:
            if not isinstance(input_tensors, tuple):
                input_tensors = (input_tensors,)

            for buffer_idx in range(len(self.input_buffers)):
                self.input_buffers[buffer_idx].copy_(input_tensors[buffer_idx], non_blocking=True)

            if target_tensors is not None:
                if not self.is_loss_supervised:
                    print("WARNING: target_tensors are ignored when loss is not supervised")
                else:
                    # TODO: check that target_tensors is not None
                    if not isinstance(target_tensors, tuple):
                        target_tensors = (target_tensors,)

                    for buffer_idx in range(len(self.target_buffers)):
                        self.target_buffers[buffer_idx].push_back(target_tensors[buffer_idx])

            self.frame_t += 1

        if not self.use_cuda_graph:
            self._step()
            self._advance_clock()
            self.main_stream.synchronize()
        else:
            self._graph.replay()
            for device in self.devices:
                torch.cuda.current_stream(device).synchronize()

        return self.output_buffers, self.frame_t - len(self.stages) - 1