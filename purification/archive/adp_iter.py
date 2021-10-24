import torch
import sys
from utils import *
from purification import *

def adp_iter(x, network_ebm, max_iter, mode, config):
    images = []
    step_sizes = []
    for i in range(config.purification.meta_max_iter):
        imgs, step_size = adp(x, network_ebm, max_iter, mode, config)
        x = imgs[-1]
        images.extend(imgs)
        step_sizes.extend(step_size)
    return images, step_sizes
