from models import *
from functions import *
from torchsummary import summary
import torch.optim as optim


def init_model():
    device = get_device()
    net = DnCNN(17, 1).to(device)
    net.apply(weights_init)
    return net


def train():
    device = get_device()
    net = init_model()
    loss_fn = net.get_lossfn()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
