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

    image_dir = "BSD"
    image_size = 180
    num_channels = 1
    batch_size = 1
    dataloader = get_dataloader(image_dir, image_size, num_channels, batch_size)

    print("Starting Training...")

    img_list = []
    losses = []
    num_cycles = 0  # training cycles (iterations)

    num_epochs = 50
    for epoch in range(num_epochs):
        for batch_ndx, data in enumerate(dataloader, 0):
            # TODO: fix noisy, target image pair in dataloader
            target_batch, _ = data
            show_image(target_batch)
            noisy_batch = corrupt_gaussian(target_batch, std=0.005)
            show_image(noisy_batch)
            break
            pass
        break
