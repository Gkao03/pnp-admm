from models import *
from functions import *
from torchsummary import summary
import torch.optim as optim
import torchvision.utils as vutils


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
            target_batch, _ = data
            noisy_batch = corrupt_gaussian(target_batch, std=0.005)

            # zero parameter gradients
            optimizer.zero_grad()

            residual = net(noisy_batch)

            loss = loss_fn(residual, noisy_batch - target_batch)
            loss.backward()
            optimizer.step()

            # Output training stats
            print('[%d/%d][%d/%d]\tLoss: %.4f' % (epoch + 1, num_epochs, batch_ndx + 1, len(dataloader), loss.item()))

            losses.append(loss.item())

            if (num_cycles % 10 == 0) or (epoch == num_epochs - 1):
                img_list.append(vutils.make_grid(residual, padding=2, normalize=True))

            num_cycles += 1

    print("Finished Training")

    PATH = "./pre_trained/dncnn.pth"
    torch.save(net.state_dict(), PATH)
