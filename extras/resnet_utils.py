from collections import OrderedDict
from typing import Iterator, Tuple

from torch import nn
from torch import Tensor
from typing import Dict, Optional, Tuple, Union

def flatten(module: nn.Sequential) -> nn.Sequential:
    """Flattens a nested sequential module."""
    if not isinstance(module, nn.Sequential):
        raise TypeError('not sequential')

    return nn.Sequential(OrderedDict(_flatten(module)))


def _flatten(module: nn.Sequential) -> Iterator[Tuple[str, nn.Module]]:
    for name, child in module.named_children():
        # Flatten child sequential layers only.
        if isinstance(child, nn.Sequential):
            for sub_name, sub_child in _flatten(child):
                yield ('%s_%s' % (name, sub_name), sub_child)
        else:
            yield (name, child)

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Twin(nn.Module):
    def forward(self,  # type: ignore
                tensor: Tensor,
                ) -> Tuple[Tensor, Tensor]:
        return tensor, tensor


class Gutter(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self,  # type: ignore
                input_and_skip: Tuple[Tensor, Tensor],
                ) -> Tuple[Tensor, Tensor]:
        input, skip = input_and_skip
        output = self.module(input)
        return output, skip


class Residual(nn.Module):
    def __init__(self, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.downsample = downsample

    def forward(self,  # type: ignore
                input_and_identity: Tuple[Tensor, Tensor],
                ) -> Tensor:
        input, identity = input_and_identity
        if self.downsample is not None:
            identity = self.downsample(identity)
        return input + identity

class Flatten(nn.Module):
    """Flattens any input tensor into an 1-d tensor."""

    def forward(self, x: Tensor):  # type: ignore
        return x.view(x.size(0), -1)



def bottleneck(inplanes: int,
               planes: int,
               stride: int = 1,
               downsample: Optional[nn.Module] = None,
               ) -> nn.Sequential:
    """Creates a bottlenect block in ResNet as a :class:`nn.Sequential`."""
    layers: Dict[str, nn.Module] = OrderedDict()
    layers['twin'] = Twin()

    layers['conv1'] = Gutter(conv1x1(inplanes, planes))
    layers['bn1'] = Gutter(nn.BatchNorm2d(planes))
    layers['conv2'] = Gutter(conv3x3(planes, planes, stride))
    layers['bn2'] = Gutter(nn.BatchNorm2d(planes))
    layers['conv3'] = Gutter(conv1x1(planes, planes * 4))
    layers['bn3'] = Gutter(nn.BatchNorm2d(planes * 4))
    layers['residual'] = Residual(downsample)
    layers['relu'] = nn.ReLU()

    return nn.Sequential(layers)