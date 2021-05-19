from models import *
from functions import *
from torchsummary import summary


def init_model():
    device = get_device()
    net = DnCNN(17, 1).to(device)
    net.apply(weights_init)
    return net


def train():
    device = get_device()
    net = init_model()
    loss_fn = net.get_lossfn()
