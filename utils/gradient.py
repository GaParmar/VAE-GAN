#https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py

import torch

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA=10):
    device = real_data.device
    batch_size = real_data.shape[0]
    real_data = real_data.view((batch_size,-1))
    fake_data = fake_data.view((batch_size,-1))
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size()).to(device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return grad_penalty

