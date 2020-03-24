
gpu_devices = "3"
random_seed = 13
num_epochs = 1
NUM_DISC_STEPS = 5

lr=1e-4
beta1 = 0.5
beta2 = 0.9

batch_size = 64

loss_names = ["recon_loss"]

dataroot = "data/mnist"
name = "mnist"
imsize=28

zdim=128

lambda_recon = 5