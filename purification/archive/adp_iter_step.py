import torch
import sys
from utils import *
from purification import *

def adp_iter_step(x, network_ebm, max_iter, mode, config):
    images = []
    step_sizes = []
    begin_sigma = config.purification.begin_sigma
    end_sigma = config.purification.end_sigma
    sigma_ls = np.logspace(np.log(begin_sigma), np.log(end_sigma), num=config.purification.meta_max_iter, base=np.exp(1))
    for i in range(config.purification.meta_max_iter):
        config.purification.rand_smoothing_level = sigma_ls[i]
        imgs, step_size = adp(x, network_ebm, max_iter, mode, config)
        x = imgs[-1]
        images.extend(imgs)
        step_sizes.extend(step_size)
    return images, step_sizes
