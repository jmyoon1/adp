import torch
import sys
from utils import *

def ddp(x, network_ebm, max_iter, mode, config):
    min_step_lr = 0.00001
    lr_min = 1.0e-3
    images = [] # From noisy initialized image to purified image
    step_sizes = [] # Step sizes

    transform_raw_to_ebm = raw_to_ebm(config.structure.dataset)
    transform_ebm_to_raw = ebm_to_raw(config.structure.dataset)

    with torch.no_grad():
        if mode=="attack":
            smoothing_level = config.attack.rand_smoothing_level
            smoothing_type = config.attack.rand_type
        else:
            smoothing_level = config.purification.rand_smoothing_level
            smoothing_type = config.purification.rand_type
        if smoothing_type=="binary":
            x_pur = torch.clamp(x + (torch.randint_like(x, 2)*2-1)*smoothing_level*np.sqrt(2.0/np.pi), 0.0, 1.0)
        else:
            x_pur = torch.clamp(x + torch.randn_like(x)*smoothing_level, 0.0, 1.0)
        x_pur = transform_raw_to_ebm(x_pur).to(config.device.ebm_device)
        images.append(x_pur.clone().detach())
        # Stopping criterion
        labels = torch.ones(x_pur.shape[0], device=x_pur.device)
        labels = labels.long().to(config.device.ebm_device)
        grad = network_ebm(x_pur, labels) # Get gradients
        x_pur = torch.clamp(transform_ebm_to_raw(x_pur+grad*smoothing_level), 0.0, 1.0)
        step_sizes.append(1.0)
        images.append(x_pur)
    return images, step_sizes
