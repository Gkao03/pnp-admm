import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DnCNN(nn.Module):
    """
    Following Denoising Architecture:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7839189
    """
    def __init__(self, depth):
        pass

    def _block(self, in_channels, out_channels, kernel_size, stride, padding, layer_type):
        block = nn.Sequential()
        block.add_module('conv', nn.ConvTranspose2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    bias=False))
        if layer_type == 2:
            block.add_module('batchnorm', nn.BatchNorm2d(out_channels))
        if layer_type == 1 or layer_type == 2:
            block.add_module('relu', nn.ReLU(inplace=True))
        return block
