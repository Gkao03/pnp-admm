import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DnCNN(nn.Module):
    def __init__(self):
        pass

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        A sequential convolutional block for PokeGAN Generator v1.
        Uses fractional-strided convolutions and ReLU activation.
        :param in_channels: num channels in input
        :param out_channels: num channels in output
        :param kernel_size: n x n size of kernel
        :param stride: stride size
        :param padding: padding size
        :return: convolutional block including convolution, batchnorm, relu
        """
        block = nn.Sequential()
        block.add_module('conv', nn.ConvTranspose2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=padding,
                                                    bias=False))
        block.add_module('batchnorm', nn.BatchNorm2d(out_channels))
        block.add_module('relu', nn.ReLU(inplace=True))
        return block