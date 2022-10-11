import argparse
from collections import OrderedDict
from multiprocessing import Pipe
from pickletools import optimize
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
import copy

import time
import torchvision
from torchvision.datasets import CIFAR10
import wandb

from common import *
from extras.networks import resnet101, resnet50


from partime.pipeline import Pipeline, DummyOptimizer
from partime.balancing import balance_pipeline_partitions


WANDB_PROJECT = "<project>"
WANDB_ENTITY = "<entity>"

DEBUG = False

LOG_EVERY = 1000

DEVICES = list(range(torch.cuda.device_count()))

class Reshape(torch.nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)

def gen_net():
    return torch.nn.Sequential(OrderedDict([
        ('conv1', torch.nn.Conv2d(3, 6, 5)),  # in, out, k, (s=1)
        ('pool1', torch.nn.MaxPool2d(2, stride=2)),
        ('conv2', torch.nn.Conv2d(6, 16, 5)),
        ('pool2', torch.nn.MaxPool2d(2, stride=2)),
        ('reshape', Reshape((-1, 16 * 5 * 5))),
        ('fc1', torch.nn.Linear(16 * 5 * 5, 120)),
        ('fc2', torch.nn.Linear(120, 84)),
        ('fc3', torch.nn.Linear(84, 10)),
    ]))

def loss_fn(outputs, labels):
    return torch.nn.functional.cross_entropy(outputs.squeeze(), labels.squeeze().to(dtype=torch.long))


def check_correct(outputs, labels):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels.squeeze()).sum().item()
    return correct


def test_seq_weights(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        correct = 0
        total = 0
        loss = 0.
        for images, labels in test_loader:
            images = images.to('cuda:0')
            labels = labels.to('cuda:0')
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss += loss_fn(outputs, labels)

    net.train()
    return correct / total, loss


def train_seq(net, train_loader, test_loader, input_shape, lr, n_epochs):
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    images_seen = 0
    batch_size = input_shape[0]
    replay_batch = torch.empty(input_shape, device='cuda:0')
    replay_labels = torch.empty(batch_size, device='cuda:0', dtype=torch.long)
    for epoch in range(n_epochs):
        for image, label in train_loader:
            image = image.to('cuda:0')
            label = label.to('cuda:0')
            index = images_seen % batch_size
            replay_batch[index] = image
            replay_labels[index] = label

            images_seen += 1
            if images_seen < batch_size - 1:
                continue

            optimizer.zero_grad()
            outputs = net(replay_batch)
            loss = loss_fn(outputs, replay_labels)
            #print(loss)
            loss.backward()
            optimizer.step()
            if images_seen % LOG_EVERY == 0:
                acc, test_loss = test_seq_weights(net, test_loader)
                if not DEBUG:
                    wandb.log({'acc': acc, 'loss': test_loss}, step=images_seen)
                print(f"Epoch {epoch}, frames seen {images_seen}, loss {test_loss}, accuracy {acc}")

    # compute test set accuracy
    final_acc, final_loss = test_seq_weights(net, test_loader)
    if not DEBUG:
        wandb.log({'final_acc': final_acc, 'final_loss': final_loss})
    print(f"Sequential test accuracy {final_acc}")


def _test_pipeline_weights(pipeline: Pipeline, test_loader):

    net = torch.nn.Sequential()

    for i, stage in enumerate(pipeline.stages):
        net.add_module(f"stage_{i}", copy.deepcopy(stage.module).to('cuda:0'))

    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        correct = 0
        total = 0
        loss = 0.
        for images, labels in test_loader:
            images = images.to('cuda:0')
            labels = labels.to('cuda:0')
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += loss_fn(outputs, labels)

    #print(f"Pipeline test accuracy {correct / total}")
    return correct / total, loss

def train_pipeline(pipeline: Pipeline, train_loader, test_loader, input_shape, n_epochs):
    images_seen = 0
    batch_size = input_shape[0]
    replay_batch = torch.empty(input_shape, device='cuda:0')
    replay_labels = torch.empty(batch_size, 1, device='cuda:0', dtype=torch.long)

    for epoch in range(n_epochs):
        labels_backups = [None] * (len(pipeline.stages) - 1)

        for image, label in train_loader:
            image = image.to('cuda:0')
            label = label.to('cuda:0')
            index = images_seen % batch_size
            replay_batch[index] = image
            replay_labels[index] = label

            images_seen += 1
            if images_seen < batch_size - 1:
                continue

            labels_backups[images_seen % len(labels_backups)] = replay_labels.cpu().clone()
            outputs = pipeline.forward(replay_batch, replay_labels.to(dtype=torch.float32))[0][0]
            #print(pipeline.loss_buffer.mean())

            if images_seen > len(pipeline.stages) - 1:
                # corresponding_labels = labels_backups[(steps_taken + 1) % len(labels_backups)]
                # correct = check_correct(outputs, replay_labels.cpu())
                # total = replay_labels.shape[0]
                if images_seen % LOG_EVERY == 0:
                    final_acc, test_loss = _test_pipeline_weights(pipeline, test_loader)
                    if not DEBUG:
                        wandb.log({'acc': final_acc, 'loss': test_loss}, step=images_seen)
                    print(f"Epoch {epoch}, frames seen {images_seen}, loss {test_loss}, accuracy {final_acc}")


    final_acc, final_loss = _test_pipeline_weights(pipeline, test_loader)
    if not DEBUG:
        wandb.log({'final_acc': final_acc, 'final_loss': final_loss})


balances = {2: [4, -1], 4: [2, 2, 2, -1]}

def main(args_cmd):

    if not DEBUG:
        run_name = f"{args_cmd.stages}_{args_cmd.lr}_{args_cmd.batch_size}"
        w_run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, config=args_cmd, name=run_name)

    torch.manual_seed(args_cmd.seed)

    seq_net = resnet50(num_classes=10).to('cuda:0')

    batch_size = args_cmd.batch_size

    input_shape = (batch_size, 3, 32, 32)

    cifar_train = torch.utils.data.DataLoader(
            CIFAR10(
                root='./cifar',
                train=True,
                download=True,
                transform=torchvision.transforms.ToTensor()
            ),
            batch_size=1,
            shuffle=True,
            drop_last=True
        )
    cifar_test = torch.utils.data.DataLoader(
        CIFAR10(
            root='./cifar',
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor()
            ),
        batch_size=256,
        shuffle=True,
        drop_last=True
    )
    if args_cmd.stages > 1:
        devices = DEVICES[:args_cmd.stages]

        balance, exp_devices = balance_pipeline_partitions(devices, seq_net, input_shape, n_partitions=args_cmd.stages)

        pipeline = Pipeline(
            seq_net,
            torch.rand(input_shape),
            balance,
            exp_devices,
            True,
            loss_fn,
            torch.rand(batch_size, 1),
            (torch.optim.Adam, {'lr': args_cmd.lr})
        )
        train_pipeline(pipeline, cifar_train, cifar_test, input_shape, args_cmd.max_epochs)
    else:
        train_seq(seq_net, cifar_train, cifar_test, input_shape, args_cmd.lr, args_cmd.max_epochs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Online learning with pipeline")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--stages', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_epochs', type=int, default=10)

    args_cmd = parser.parse_args()

    main(args_cmd)