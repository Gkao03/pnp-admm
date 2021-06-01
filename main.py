from denoise import *
from models import *
from os import path


def main():
    dncnn_path = "pre_trained/dncnn512.pth"
    device = get_device()

    if not path.exists(dncnn_path):
        losses, img_list = train()

    net = DnCNN(17, 1)
    net.load_state_dict(torch.load(dncnn_path, map_location=device))
    net.eval()

    target = image2tensor_norm('images/barbara.png').unsqueeze(0)
    noisy = corrupt_gaussian(device, target, std=0.005)
    residual = net(noisy)
    x = noisy - residual


if __name__ == '__main__':
    main()
