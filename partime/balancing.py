"""Collection of functions for balancing partitions"""

from operator import itemgetter
import torch
from typing import List, Tuple, Union
from enum import Enum

from zmq import device

def _normalize_devices(devices: List[Union[torch.device, int]]):
    """
    Normalizes the list of devices to a list of torch.device objects.
    """
    return [
        torch.device(f'cuda:{device}') if isinstance(device, int) else device
        for device in devices
    ]

class Direction(Enum):
    H2D = "h2d"
    D2H = "d2h"

def _measure_copy_time(src_device: torch.device, dst_device: torch.device, buf_size: torch.Size, direction: Direction, n_trials: int = 20, n_warmup_trials: int = 2):
    """
    Measures the time of a copy from src_device to dst_device.
    """
    src_buf = torch.rand(buf_size, device=src_device)
    dst_buf = torch.rand(buf_size, device=dst_device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    measures = []

    cuda_device = dst_device if direction == Direction.H2D else src_device

    for trial in range(-n_warmup_trials, n_trials):
        torch.cuda.synchronize(cuda_device)
        start_event.record()

        dst_buf.copy_(src_buf)

        end_event.record()
        torch.cuda.synchronize(cuda_device)

        if trial >= 0:
            transfer_time = start_event.elapsed_time(end_event)
            measures.append(transfer_time)

    return measures

def _compute_throughput(time_ms: float, buf_size_mb: int):
    time_s = time_ms / 1000.
    buf_size_gb = buf_size_mb / 1024.
    return buf_size_gb / time_s

def _sequence_block_partitions(operation_times: List[float], n_partitions: int):
    """
    Divides the sequence of timings in n_partitions blocks of sequential operations,
    each block having similar total_times to other blocks.
    """

    if n_partitions < 1:
        return ValueError("n_partitions must be greater than 0")

    if len(operation_times) <= n_partitions:
        return ValueError("n_partitions must be smaller than the number of timings")


    # normalize operation_times to [0, 1]
    normalized_times = [
        (time - min(operation_times)) / (max(operation_times) - min(operation_times))
        for time in operation_times
    ]

    # from here we apply the algorithm in https://arxiv.org/abs/1308.2452.pdf
    # it takes O(n_partitions * len(operation_times)^3) operations

    # create the arbitrary initial split of the sequence.
    # This initializes the split into blocks of even number of operations.
    # split is the variable P in the paper.
    n = len(normalized_times)
    split = [n//n_partitions * (x+1) for x in range(n_partitions-1)] + [n]

    def get_blocks_total_time(times):
        indices = [0] + split
        return [
            (sum(times[indices[i]:indices[i+1]]), i)
            for i in range(n_partitions)
        ]

    # (1) Fix p ∈ [k] with M(P) = b_p. So B_p is a maximal block of P
    # in the paper, max_idx is p, max_P is M(P)
    max_time, max_idx = max(get_blocks_total_time(normalized_times), key=itemgetter(0))

    while True:
        # in the paper, min_idx is q, min_P is m(P)
        min_time, min_idx = min(get_blocks_total_time(normalized_times), key=itemgetter(0))

        # (2) If M(P) ≤ m(P) + 1, then stop.
        if max_time <= min_time + 1.:
            break

        # (3) If M(P) > m(P) + 1, we update the block closer to min_P in the direction of the max_P

        # either max_idx < min_idx or max_idx > min_idx
        # they cannot be the same or the previous if would have been true
        if max_idx < min_idx:
            # this means that the smallest partition is on the right of the biggest partition
            h = min_idx - 1
            split[h] -= 1 # we give the smallest partition the last operation from h
        else:
            h = min_idx + 1
            split[min_idx] += 1 # we give the smallest partition the first operation from h

        if h == max_idx:
            # we changed the biggest partition, so we need to recompute max_P and max_idx
            max_time, max_idx = max(get_blocks_total_time(normalized_times), key=itemgetter(0))

    # we return the number of operations in each partition
    indices = [0] + split
    balance =  [
        (indices[i+1] - indices[i])
        for i in range(n_partitions)
    ]

    return balance, [t[0] for t in get_blocks_total_time(normalized_times)], [t[0] for t in get_blocks_total_time(operation_times)]

def benchmark_cpu_to_gpu_copy(devices: List[Union[torch.device, int]], sample_input_size: torch.Size, n_trials = 20, n_warmup_trials = 2):
    """
    Measures the speed of transfer from CPU to each GPU in the list.
    Returns a dict with the device name as key and the speed in GB/s as value.

    Args:
        devices: List of devices to measure.
        sample_input_size: The size of a sample input.
        n_trials: Number of copies to evaluate average throughput. Defaults to 20.
    """

    devices = _normalize_devices(devices)
    measures = { device.index: [] for device in devices }

    for device in devices:
        with torch.cuda.device(device):
            measures[device.index] = _measure_copy_time(
                torch.device('cpu'),
                device,
                sample_input_size,
                Direction.H2D,
                n_trials,
                n_warmup_trials
            )

    return {
        device_id: sum(measure_list)/len(measure_list)
        for device_id, measure_list in measures.items()
    }

def benchmark_gpu_to_cpu_copy(devices: List[Union[torch.device, int]], sample_input_size: torch.Size, n_trials = 20, n_warmup_trials = 2):
    """
    Measures the speed of transfer from each GPU in the list to CPU.
    Returns a dict with the device name as key and the speed in GB/s as value.

    Args:
        devices: List of devices to measure.
        sample_input_size: The size of a sample input.
        n_trials: Number of copies to evaluate average throughput. Defaults to 20.
    """

    devices = _normalize_devices(devices)
    measures = { device.index: [] for device in devices }

    for device in devices:
        with torch.cuda.device(device):
            measures[device.index] = _measure_copy_time(
                device,
                torch.device('cpu'),
                sample_input_size,
                Direction.D2H,
                n_trials,
                n_warmup_trials
            )

    return {
        device_id: sum(measure_list)/len(measure_list)
        for device_id, measure_list in measures.items()
    }


def benchmark_gpu_to_gpu_copy_speed(devices: List[torch.device], sample_input_size: torch.Size, n_trials = 20, n_warmup_trials = 2):
    """
    Measures the speed of transfer from one GPU to another.
    Returns a dict with a tuple of the two devices indices as key and the speed in GB/s as value.

    Args:
        devices: List of devices to measure.
        buf_size_mb: Size in MiB of the buffer to transfer. Defaults to 128MiB.
        n_trials: Number of copies to evaluate average throughput. Defaults to 20.
    """
    devices = _normalize_devices(devices)

    device_pairs = [(device1, device2) for device1 in devices for device2 in devices if device1 != device2]

    buffers = {
        device.index: torch.rand(sample_input_size, device=device)
        for device in devices
    }

    measures = {
        (device1.index, device2.index): []
        for device1, device2 in device_pairs
    }

    # initialize events for measuring copy time

    for src_device, dst_device in device_pairs:
        key = (src_device.index, dst_device.index)
        with torch.cuda.device(src_device):
            measures[key] = _measure_copy_time(
                src_device,
                dst_device,
                sample_input_size,
                n_trials,
                n_warmup_trials
            )

    return {
        pair_key: sum(measure_list)/len(measure_list)
        for pair_key, measure_list in measures.items()
    }

def benchmark_sequential_network(devices: List[Union[torch.device, int]], network: torch.nn.Sequential, sample_input_size: torch.Size, n_trials: int = 20, n_warmup_trials: int = 2):
    """
    Measures the speed of sequential computation on each GPU in the list.
    Returns a dict with a tuple of device index and layer name as key and the speed in GB/s as value.

    Args:
        devices: List of devices to measure.
        input_res: Resolution of input image.
        n_trials: Number of copies to evaluate average throughput. Defaults to 20.
    """
    devices = _normalize_devices(devices)
    layers = list(network.named_children())

    # dict with measures for each pair device-layer
    measures = {
        layer_name: {
            device.index: [] for device in devices
        }
        for layer_name, _ in layers
    }

    for device in devices:
        with torch.cuda.device(device):
            # initialize events for measuring copy time
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            # initialize current_input with the given resolution
            current_input = torch.rand((sample_input_size), device=device)

            for layer_name, layer in layers:
                layer = layer.to(device)
                for trial in range(-n_warmup_trials, n_trials):
                    torch.cuda.synchronize(device)
                    start_event.record()

                    output = layer(current_input)

                    end_event.record()
                    torch.cuda.synchronize(device)

                    if trial >= 0:
                        process_time = start_event.elapsed_time(end_event)
                        measures[layer_name][device.index].append(process_time)

                # save input size in MiB
                current_input = output

    # take the avg for both devices
    avg_measures = {
        layer_name: {
            device_idx: sum(measure_list)/len(measure_list)
            for device_idx, measure_list in measures_by_device.items()
        }
        for layer_name, measures_by_device in measures.items()
    }

    return {
        layer_name: max(avg_measure.values())
        for layer_name, avg_measure in avg_measures.items()
    }

def balance_pipeline_partitions(devices: List[Union[torch.device, int]], network: torch.nn.Sequential, sample_input_size: torch.Size, n_trials: int = 20, n_warmup_trials: int = 2, n_partitions = None):

    devices = _normalize_devices(devices)
    net_device = list(network.children())[0].weight.device
    sample_input = torch.rand((sample_input_size), device=net_device)
    sample_output = network(sample_input)

    host2device_times = benchmark_cpu_to_gpu_copy(devices, sample_input_size=sample_input_size, n_trials=n_trials, n_warmup_trials=n_warmup_trials)
    device2host_speed = benchmark_gpu_to_cpu_copy(devices, sample_input_size=sample_output.size(), n_trials=n_trials, n_warmup_trials=n_warmup_trials)
    times = benchmark_sequential_network(devices, network, sample_input_size=sample_input_size, n_trials=n_trials, n_warmup_trials=n_warmup_trials)

    # we take the gpu with lowest host2device time as the first device
    first_device_id = min(host2device_times, key=host2device_times.get)

    # we take the gpu with lowest device2host time as the last device
    last_device_id = min(device2host_speed, key=device2host_speed.get)

    # if first and last device are the same, just take the next one in order
    if last_device_id == first_device_id:
        device2host_speed.pop(last_device_id)
        last_device_id = min(device2host_speed, key=device2host_speed.get)

    sorted_devices = _normalize_devices(
        [first_device_id] +
        [device for device in devices if device.index not in [first_device_id, last_device_id]] +
        [last_device_id]
    )

    # create the sequence of operations [h2d, layers, d2h], so that they can be split in even partitions
    operation_times = [
        host2device_times[first_device_id],
        *times.values(),
        host2device_times[last_device_id]
    ]

    # split the operations in even partitions

    n_partitions = n_partitions if n_partitions is not None else len(sorted_devices)
    balance, _, _ = _sequence_block_partitions(operation_times, n_partitions)

    # strip the copies from the balance, since we need to know how to split layers in the network
    balance[0] -= 1
    balance[1] -= 1

    return balance, sorted_devices
