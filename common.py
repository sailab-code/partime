from typing import Sequence
import pandas as pd
import torch

import time
import copy

from partime.pipeline import DummyOptimizer, Pipeline
from partime.balancing import balance_pipeline_partitions

def create_optimizer(net, optimizer_settings):
    return optimizer_settings[0](net.parameters(), **optimizer_settings[1])

def run_vanilla(network: torch.nn.Module, new_input: torch.Tensor, device: torch.device, loss_fn = None, optimizer: torch.optim.Optimizer = None, static_input_area: torch.Tensor = None, static_output_area: torch.Tensor = None, zero_grads = False):
    only_forward = loss_fn is None


    if static_input_area is None:
        input_area = new_input.to(device)
    else:
        input_area = static_input_area
        input_area.copy_(new_input, non_blocking=True)
    output = network(input_area)

    if not only_forward:
        if optimizer is not None and zero_grads:
            optimizer.zero_grad()

        loss_v = loss_fn(output)
        loss_v.backward()

        if optimizer is not None:
            optimizer.step()

    if static_output_area is None:
        output_area = output.to(torch.device('cpu'), non_blocking=True)
    else:
        output_area = static_output_area
        output_area.copy_(output_area, non_blocking=True)

    return output_area

def run_pipeline(pipeline, input: torch.Tensor):
    out = pipeline.forward(input)
    return out

def compute_naive_balance(seq_net, splits=1):
    """Splits a given network into multiple layer partitions (last partitions might be larger than the first ones)."""
    num_children = len(seq_net)
    balance = [None] * splits
    p = 0
    qq = 0
    q = int(num_children) // int(splits)
    r = int(num_children) % int(splits)

    # recollecting original modules into multiple sequential containers
    for mod_name, mod in seq_net.named_children():
        if balance[p] is None:
            qq = q + (1 if r > 0 else 0)
            balance[p] = 0
        if qq > 0:
            balance[p] += 1
            qq -= 1
        if qq == 0:
            r -= 1 if r > 0 else 0
            p += 1

    return balance


def get_random_tensor(resolution, device):
    return torch.rand(1, 3, resolution, resolution, device=device)

def dummy_loss_fn(t):
    return torch.mean(t) ** 2

def measure_vanilla_times_no_graph(net: torch.nn.Module, stream: Sequence[torch.Tensor], device: torch.device, loss_fn = None, optimizer = None):
    times = []
    for frame in stream:
        start = time.time()
        run_vanilla(net, frame, device, loss_fn, optimizer)
        torch.cuda.current_stream(device).synchronize()
        end = time.time()
        times.append(end - start)

    return times

def measure_vanilla_times_graph(net: torch.nn.Module, stream: Sequence[torch.Tensor], device: torch.device, loss_fn = None, optimizer = None):

    do_backward = loss_fn is not None

    times = []

    static_frame_area = torch.rand_like(stream[0], device='cpu', pin_memory=True)
    static_input_area = torch.rand_like(stream[0], device=device)
    static_output_area = torch.rand_like(net(stream[0].to(device)), device=device)

    if optimizer is None:
        optimizer = DummyOptimizer(net.parameters())

    graph = torch.cuda.graphs.CUDAGraph()
    main_stream = torch.cuda.Stream(device)

    # warmup
    main_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(main_stream):
        for _ in range(3):
            if do_backward:
                optimizer.zero_grad(set_to_none=True)
            run_vanilla(net, static_frame_area, device, loss_fn, optimizer, static_input_area, static_output_area, False)

    torch.cuda.current_stream(device).wait_stream(main_stream)

    # recording
    if do_backward:
        optimizer.zero_grad(set_to_none=True)

    with torch.cuda.graph(graph):
        run_vanilla(net, static_frame_area, device, loss_fn, optimizer, static_input_area, static_output_area, False)

    torch.cuda.current_stream(device).synchronize()

    for frame in stream:
        static_frame_area.copy_(frame)
        torch.cuda.current_stream(device).synchronize()
        start = time.time()
        graph.replay()
        torch.cuda.current_stream(device).synchronize()
        end = time.time()
        times.append(end - start)

    return times


def measure_pipeline_times(pipeline: Pipeline, stream: Sequence[torch.Tensor]):
    times = []
    for frame in stream:
        start = time.time()
        run_pipeline(pipeline, frame)
        end = time.time()
        times.append(end - start)

    return times

def run_measurements(net, stream, device, optimizer_settings, devices, forward_only = False, naive_balance=True):
    net_bkp = net

    # run base experiment with clean network

    vanilla_times = {
        'no_graph': {
            'with_backward': None,
            'only_forward': None
        },
        'with_graph': {
            'with_backward': None,
            'only_forward': None
        }
    }

    # run experiment with backward and with cuda graph

    if not forward_only:
        net = copy.deepcopy(net_bkp)
        vanilla_times['with_graph']['with_backward'] = measure_vanilla_times_graph(net, stream, device, dummy_loss_fn, create_optimizer(net, optimizer_settings))

        # run experiment with backward and without cuda graph

        net = copy.deepcopy(net_bkp)
        vanilla_times['no_graph']['with_backward'] = measure_vanilla_times_no_graph(net, stream, device, dummy_loss_fn, create_optimizer(net, optimizer_settings))

    # run experiments without backward and with cuda graph

    net = copy.deepcopy(net_bkp)
    vanilla_times['with_graph']['only_forward'] = measure_vanilla_times_graph(net, stream, device, None, None)

    # run experiment without backward and without cuda graph

    net = copy.deepcopy(net_bkp)
    vanilla_times['no_graph']['only_forward'] = measure_vanilla_times_no_graph(net, stream, device, None, None)

    # run pipeline experiments

    pipeline_times_by_device = {}

    for n_stages in range(2, len(devices) + 1):
        print(f"Running pipeline experiment with {n_stages} devices...")

        if naive_balance:
            balance, exp_devices = compute_naive_balance(net, n_stages), devices[:n_stages]
        else:
            balance, exp_devices = balance_pipeline_partitions([i for i in range(n_stages)], net, stream[0].shape)

        pipeline_times = {
            'no_graph': {
                'with_backward': None,
                'only_forward': None
            },
            'with_graph': {
                'with_backward': None,
                'only_forward': None
            }
        }

        if not forward_only:
            # run experiment with backward and cuda graph
            pipeline = Pipeline(
                net,
                stream[0],
                balance[0],
                exp_devices,
                True,
                dummy_loss_fn,
                optimizer_settings
            )
            pipeline_times['with_graph']['with_backward'] = measure_pipeline_times(pipeline, stream)

            #run experiment with backward and without cuda_graph
            pipeline = Pipeline(
                net,
                stream[0],
                balance,
                exp_devices,
                False,
                dummy_loss_fn,
                optimizer_settings
            )
            pipeline_times['no_graph']['with_backward'] = measure_pipeline_times(pipeline, stream)

        # run experiment without backward and cuda graph
        pipeline = Pipeline(
            net,
            stream[0],
            balance,
            exp_devices,
            True,
            None,
            None
        )
        pipeline_times['with_graph']['only_forward'] = measure_pipeline_times(pipeline, stream)

        # run experiment without backward and without cuda graph
        pipeline = Pipeline(
            net,
            stream[0],
            balance,
            exp_devices,
            False,
            None,
            None
        )
        pipeline_times['no_graph']['only_forward'] = measure_pipeline_times(pipeline, stream)

        pipeline_times_by_device[n_stages] = pipeline_times


    df_rows = []
    if not forward_only:
        keypairs = [
            ('no_graph', 'with_backward'),
            ('no_graph', 'only_forward'),
            ('with_graph', 'with_backward'),
            ('with_graph', 'only_forward')
        ]
    else:
        keypairs = [
            ('no_graph', 'only_forward'),
            ('with_graph', 'only_forward')
        ]

    # add vanilla rows
    vanilla_rows = [
        {
            'time': t,
            'pipeline': False,
            'n_devices': 1,
            'cuda_graph': keypair[0] == 'with_graph',
            'backward': keypair[1] == 'with_backward',
        }
        for keypair in keypairs
        for t in vanilla_times[keypair[0]][keypair[1]]
    ]

    df_rows += vanilla_rows

    #add pipeline rows

    pipeline_rows = []
    for n_stages, pipeline_times in pipeline_times_by_device.items():
        for keypair in keypairs:
            for t in pipeline_times[keypair[0]][keypair[1]]:
                pipeline_rows.append({
                    'time': t,
                    'pipeline': True,
                    'n_devices': int(n_stages),
                    'cuda_graph': keypair[0] == 'with_graph',
                    'backward': keypair[1] == 'with_backward',
                })

    df_rows += pipeline_rows

    df = pd.DataFrame(df_rows)

    return df