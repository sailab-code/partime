from pickletools import optimize
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
import copy

import time

from common import *


from partime.pipeline import Pipeline, DummyOptimizer

resolution_settings = {
    'i': 256,
    'ii': 1024
}

output_settings = {
    'a': 1,
    'b': 10,
    'c': 100
}

input_channels = 3
hidden_features = 32
padding = 2
layers = 30 # 150
kernel_size = 5
stream_size = 5 # 1500
batch_size = 1

optimizer_settings = (torch.optim.Adam, {'lr': 0.01})

n_devices = min(2, torch.cuda.device_count())
devices = list(range(0, n_devices))

now = datetime.now()
base_results_path = Path(f"./results/experimentA/{now.strftime('%y-%m-%d_%H-%M-%S')}")
base_results_path.mkdir(parents=True, exist_ok=True)

def generate_network(layers_count, input_channels = 3, hidden_channels = 32, output_channels = 3, kernel_size = 5, padding = 2):
    """
    Generate a network with the given number of layers and channels
    """

    # Create the network
    network = torch.nn.Sequential()

    # Add the first layer
    network.add_module("input_layer", torch.nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=padding))
    network.add_module("relu_1", torch.nn.ReLU())

    # Add the rest of the layers
    for i in range(layers_count-1):
        network.add_module("conv_layer_" + str(i+2), torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=1, padding=padding))
        network.add_module("relu_" + str(i+2), torch.nn.ReLU())

    # Add the output layer
    network.add_module("output_layer", torch.nn.Conv2d(hidden_channels, output_channels, kernel_size=kernel_size, stride=1, padding=padding))

    # Add sigmoid layer
    network.add_module("output_sigmoid", torch.nn.Sigmoid())

    return network

def subexperiment(resolution_option, output_option):
    resolution = resolution_settings[resolution_option]
    output_channels = output_settings[output_option]
    print(f"Running experiments for resolution {resolution} and output channels {output_channels}...")
    out_path = Path(base_results_path, f'{resolution_option}_{output_option}.csv')

    print(f"Running base experiment...")
    net = generate_network(layers, input_channels, hidden_features, output_channels, kernel_size, padding)
    net.to(devices[0])

    stream = [ get_random_tensor(resolution, 'cpu') for _ in range(stream_size) ]

    device = torch.device('cuda:0')

    df = run_measurements(net, stream, device, optimizer_settings, devices, forward_only=False)
    df.to_csv(out_path, index=False)


def main():
    for resolution_option in resolution_settings:
        for output_option in output_settings:
            subexperiment(resolution_option, output_option)

if __name__ == '__main__':
    main()