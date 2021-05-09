import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def image2tensor(image_path):
    im = Image.open(image_path)
    np_array = np.asarray(im).astype("float32")
    tensor = torch.from_numpy(np_array)
    return tensor


def image2tensor_norm(image_path):
    im = Image.open(image_path)
    tensor = transforms.ToTensor()(im)
    return tensor


def corrupt_gaussian(tensor):
    pass
