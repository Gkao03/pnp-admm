import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
import os
import random


def image2tensor(image_path):
    im = Image.open(image_path)
    np_array = np.asarray(im).astype("float32")
    tensor = torch.from_numpy(np_array)
    return tensor


def image2tensor_norm(image_path):
    im = Image.open(image_path)
    tensor = transforms.ToTensor()(im)
    return tensor


def corrupt_gaussian(tensor, std=1):
    noise = torch.randn(tensor.size()) * std**0.5
    corrupted_tensor = tensor + noise
    return corrupted_tensor


def show_image(tensor):
    im = tensor
    if len(tensor.size()) == 4:
        im = torchvision.utils.make_grid(tensor)

    npimg = im.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def weights_init(m):
    """
    weights initialization for netG and netD
    :param m: model
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def transform_image_function(image_size, num_channels):
    """
    Returns a function that defines how the input images are transformed
    :param image_size: n x n size of the image
    :param num_channels: number of channels (3 for color)
    :return: the transform function
    """
    list_of_transforms = [
        transforms.Resize(image_size),  # square resize
        transforms.CenterCrop(image_size),  # square crop
        transforms.ToTensor(),  # convert image to pytorch tensor
    ]
    transform = transforms.Compose(list_of_transforms)
    return transform


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def get_dataset(dataroot, image_size, num_channels):
    transform = transform_image_function(image_size, num_channels)
    dataset = dset.ImageFolder(root=dataroot, transform=transform)
    return dataset


def get_dataloader(dataroot, image_size, num_channels, batch_size):
    dataset = get_dataset(dataroot, image_size, num_channels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader
