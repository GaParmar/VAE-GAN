# Model definition borrowed from here
# https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, DIM=64, zdim=512):
        super(Encoder, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*DIM, zdim*2)
        self.DIM = DIM

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        batch_size = input.shape[0]
        out = self.main(input)
        out = out.view(-1, 4*4*4*self.DIM)
        out = self.output(out)
        return out.view(batch_size, -1)

class Generator(nn.Module):
    def __init__(self, DIM=64):
        super(Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4*DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()
        self.DIM = DIM

    def forward(self, input):
        batch_size = input.shape[0]
        output = self.preprocess(input)
        output = output.view(-1, 4*self.DIM, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        return output.view(batch_size, -1)
    

class Discriminator(nn.Module):
    def __init__(self, DIM=64):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*DIM, 1)
        self.DIM = DIM

    def forward(self, input, output_logits=False):
        input = input.view(-1, 1, 28, 28)
        output = self.main(input)
        if not output_logits:
            output = output.view(-1, 4*4*4*self.DIM)
            output = self.output(output)
            return output
        else:
            output_logits = output.view(-1, 4*4*4*self.DIM)
            return output_logits