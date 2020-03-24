import os
import sys
import pdb
import importlib

from networks.models_28x28 import *
from utils.common import *
from utils.gradient import *
from utils.visualizations import *

exp_name = sys.argv[1] + "_baseline2"
C = importlib.import_module(f"configs.{sys.argv[1]}")

# import the models based on image resolution
fname = f"networks.models_{C.imsize}x{C.imsize}"
M = importlib.import_module(fname)

set_cuda_devices(vals=C.gpu_devices)
disable_warnings()
set_seed(seed=C.random_seed)
device = torch.device("cuda")

netE = M.Encoder(zdim=C.zdim).to(device)
netG = M.Generator().to(device)
netD = M.Discriminator().to(device)
optimizerE = torch.optim.Adam(netE.parameters(), lr=C.lr, betas=(C.beta1, C.beta2))
optimizerD = torch.optim.Adam(netD.parameters(), lr=C.lr, betas=(C.beta1, C.beta2))
optimizerG = torch.optim.Adam(netG.parameters(), lr=C.lr, betas=(C.beta1, C.beta2))

DL = get_dataloader(name=C.name, dataroot=C.dataroot, batch_size=C.batch_size, imsize=C.imsize)

log_file = f"output/EXP_LOGS/log_{exp_name}.txt"
if not os.path.exists(f"output/EXP_LOGS"):
    os.makedirs(f"output/EXP_LOGS")
if not os.path.exists(log_file):
    with open(log_file, 'w'): pass
if not os.path.exists(f"output/VIZ/{exp_name}"):
    os.makedirs(f"output/VIZ/{exp_name}")
if not os.path.exists(f"output/saved_models/{exp_name}"):
    os.makedirs(f"output/saved_models/{exp_name}")

# initialize the loss dict to empty lists
L = {}
for n in C.loss_names:
    L[n] = []

for epoch in range(C.num_epochs):
    for idx, batch_data in enumerate(DL, 0):
        batch_size = batch_data[0].shape[0]
        for d_iter in range(C.NUM_DISC_STEPS):
            x_in = batch_data[0].to(device)
            batch_size = x_in.shape[0]
            netD.zero_grad()
            # train with real
            D_real = netD(x_in).mean()
            label = torch.ones((batch_size)).to(device)*-1
            # D_real.backward(label)
            # train with fake
            noise = torch.randn(batch_size, 128).to(device)
            fake = netG(noise)
            D_fake = netD(fake.detach()).mean()
            label = torch.ones((batch_size)).to(device)
            # D_fake.backward(label)
            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, x_in, fake)
            # gradient_penalty.backward()
            D_cost = D_fake - D_real + gradient_penalty
            D_cost.backward()
            Wasserstein_D = D_real - D_fake
            optimizerD.step()
        
        
        # (2) Update G network
        netG.zero_grad()
        noise = torch.randn(batch_size, 128).to(device)
        fake = netG(noise)
        G = netD(fake).mean()
        label = torch.ones((batch_size)).to(device)
        # G.backward(label)
        G_cost = -G
        G_cost.backward()
        optimizerG.step()

        ## (3) reconstruction loss
        optimizerG.zero_grad()
        optimizerE.zero_grad()
        mu_logvar = netE(x_in)
        mu, logvar = mu_logvar[:,0:C.zdim], mu_logvar[:,C.zdim:]
        z = reparametrize(mu, logvar)
        x_rec = netG(z)
        feat_x_rec = netD(x_rec, True)
        feat_x_in = netD(x_in, True)
        recon_loss = torch.nn.MSELoss(reduction="sum")\
                            (feat_x_rec.view(batch_size,-1),\
                             feat_x_in.view(batch_size,-1))*C.lambda_reconrecon_loss.backward()
        optimizerE.step()
        optimizerG.step()


        # the logs
        log = f"{epoch:03d}-({idx}): "
        for n in C.loss_names:
            val = eval(n).item()
            log += f"{n}: {val:.3f}  "
            L[n].append(val)
        print(log)
        with open(log_file, "a") as f:
            f.write("\n"+log)
        
        if idx%100 == 0:
            disp_images(fake[0:20], f"output/VIZ/{exp_name}/{epoch}_{idx}_fake.png", 5, imsize=C.imsize)
            # recon image 
            ip = x_in[0:10]
            rec = x_rec[0:10].view(10,-1,C.imsize,C.imsize)
            disp_images(torch.cat((ip,rec),dim=0), f"output/VIZ/{exp_name}/{epoch}_{idx}_recon.png", 10, imsize=C.imsize)


    # save model
    if (epoch % 10) == 0:
        path = f"output/saved_models/{exp_name}/"
        torch.save(netG.state_dict(), path+f"netG_{epoch}.pth")
        torch.save(netD.state_dict(), path+f"netD_{epoch}.pth")