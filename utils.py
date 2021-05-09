import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

image_path = "images/barbara.png"


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
    plt.figure()
    plt.imshow(tensor.numpy().transpose(1, 2, 0), cmap='gray')
    plt.show()
