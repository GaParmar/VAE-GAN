import os
import sys
import pdb
import importlib

from networks.models_28x28 import *
from utils.common import *
from utils.gradient import *
from utils.visualizations import *

exp_name = "baseline1_"+sys.argv[1]
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
        x_in = batch_data[0].to(device)
        

        ## (3) reconstruction loss
        optimizerG.zero_grad()
        optimizerE.zero_grad()
        optimizerD.zero_grad()
        mu_logvar = netE(x_in)
        mu, logvar = mu_logvar[:,0:C.zdim], mu_logvar[:,C.zdim:]
        z = reparametrize(mu, logvar)
        x_rec = netG(z)
        feat_x_rec = netD(x_rec, True)
        feat_x_in = netD(x_in, True)
        recon_loss = torch.nn.MSELoss(reduction="sum")\
                            (feat_x_rec.view(batch_size,-1),\
                             feat_x_in.view(batch_size,-1))*C.lambda_recon
        recon_loss.backward()
        optimizerE.step()
        optimizerG.step()
        optimizerD.step()

        with torch.no_grad():
            noise = torch.randn(batch_size, C.zdim).to(device)
            fake = netG(noise)


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
        torch.save(netE.state_dict(), path+f"netE_{epoch}.pth")
        torch.save(netG.state_dict(), path+f"netG_{epoch}.pth")
        torch.save(netD.state_dict(), path+f"netD_{epoch}.pth")