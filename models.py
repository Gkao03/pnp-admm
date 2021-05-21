import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DnCNN(nn.Module):
    """
    Following Denoising Architecture:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7839189
    """
    def __init__(self, depth, in_channels):
        super(DnCNN, self).__init__()
        self.main = nn.Sequential()
        self.main.add_module('block1', self._block(in_channels, 64, 3, 1, 1, 1))  # first layer type

        for i in range(depth - 2):  # second layer type
            block_name = 'block' + str(i + 2)
            self.main.add_module(block_name, self._block(64, 64, 3, 1, 1, 2))

        block_name = 'block' + str(depth)
        self.main.add_module(block_name, self._block(64, in_channels, 3, 1, 1, 3))  # third layer type

    @staticmethod
    def _block(in_channels, out_channels, kernel_size, stride, padding, layer_type):
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

    def forward(self, x):
        return self.main(x)

    # @staticmethod
    # def _avgMSELoss(x, y, R_y):
    #     batch_size = R_y.size()[0]
    #     mse = torch.square(torch.norm(R_y - (y - x)))
    #     loss = (1 / (2 * batch_size)) * mse
    #     return loss

    def get_lossfn(self):
        return 0.5 * nn.MSELoss()
