# VAE-GAN
PyTorch naive combining of training WGAN-GP with reconstruction loss

# Usage
- `python3 train.py <exp_name>`
- Config file must be provided at `./configs/<exp_name>.py`
- experiment logs saved in `./EXP_LOGS/log_<exp_name>.txt`

# Example
- `python3 train.py baseline_mnist_vaegan`