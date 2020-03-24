import torch
import torchvision
import numpy as np

def disp_images(img, fname, nrow, imsize=28):
    batch_size = img.shape[0]
    img = img.view(batch_size,-1,imsize,imsize)
    img_ = (img.detach().cpu())
    grid =  torchvision.utils.make_grid(img_,nrow=nrow)
    torchvision.utils.save_image(grid, fname) 