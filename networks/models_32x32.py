# Model definition borrowed from here
# https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, DIM=128, zdim=128):
        super(Encoder, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )
        self.DIM=DIM
        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 2*zdim)

    def forward(self, input):
        batch_size = input.shape[0]
        input = input.view(-1, 3, 32, 32)
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.DIM)
        output = self.linear(output)
        return output.view(batch_size,-1)

class Generator(nn.Module):
    def __init__(self, DIM=64):
        super(Generator, self).__init__()
        preprocess = nn.Sequential(
            nn.Linear(128, 4 * 4 * 4 * DIM),
            nn.BatchNorm1d(4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 2, stride=2),
            nn.BatchNorm2d(2 * DIM),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 2, stride=2),
            nn.BatchNorm2d(DIM),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()
        self.DIM=DIM

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, 3, 32, 32)


class Discriminator(nn.Module):
    def __init__(self, DIM=128):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(3, DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(DIM, 2 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * DIM, 4 * DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )
        self.DIM=DIM
        self.main = main
        self.linear = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        input = input.view(-1, 3, 32, 32)
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.DIM)
        output = self.linear(output)
        return output
