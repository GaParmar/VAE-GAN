
gpu_devices = "1"
random_seed = 13
num_epochs = 200
NUM_DISC_STEPS = 5

lr=1e-4
beta1 = 0.5
beta2 = 0.9

batch_size = 64

loss_names = ["D_cost", "Wasserstein_D", "G_cost", "recon_loss"]

dataroot = "data/cifar10"
name="cifar10"
imsize=32

zdim=128

lambda_recon = 5
