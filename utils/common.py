import os
import random
from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def set_cuda_devices(vals="0,1,2,3"):
    os.environ["CUDA_VISIBLE_DEVICES"]=vals

def disable_warnings():
    # ignore the warnings
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

def set_seed(seed=13):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + std*eps

def get_dataloader(name, dataroot, batch_size, imsize=28):
    if name == "mnist":
        dataset = torchvision.datasets.MNIST(root=dataroot,
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize((imsize, imsize)),
                        transforms.ToTensor(),
                    ]))
    elif name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=dataroot,
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize((imsize, imsize)),
                        transforms.ToTensor(),
                    ]))

    DL = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=True, num_workers=8,
                    drop_last=True)
    return DL