import torch
import sys
from utils import *

def adp(x, network_ebm, max_iter, mode, config):
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
            x_pur = torch.clamp(x + (torch.randint_like(x, 2)*2-1)*smoothing_level*np.sqrt(2.0/3.1415), 0.0, 1.0)
        else:
            x_pur = torch.clamp(x + torch.randn_like(x)*smoothing_level, 0.0, 1.0)
        x_pur = transform_raw_to_ebm(x_pur).to(config.device.ebm_device)
        images.append(x_pur.clone().detach())
        cont_purification = torch.ones(x_pur.shape[0], dtype=torch.bool).to(config.device.ebm_device)
        # Stopping criterion
        for i in range(max_iter):
            labels = torch.ones(x_pur.shape[0], device=x_pur.device)
            labels = labels.long().to(config.device.ebm_device)
            grad = network_ebm(x_pur, labels) # Get gradients
            # Get adaptive step size
            x_eps = x_pur + lr_min*grad
            print(torch.mean(torch.norm(grad.view(grad.shape[0],-1), p=2,dim=1)).item(), flush=True)
            grad_eps = network_ebm(x_eps, labels)
            z1 = torch.bmm(grad.view(grad.shape[0], 1, -1), grad_eps.view(grad_eps.shape[0], -1, 1))
            z2 = torch.bmm(grad.view(grad.shape[0], 1, -1), grad.view(grad.shape[0], -1, 1))
            z = torch.div(z1, z2)
            if mode=="attack":
                step_lambda = config.attack.attack_lambda
            elif mode=="purification":
                step_lambda = config.purification.purification_lambda
            else:
                sys.exit(0)
            step_size = torch.clamp(step_lambda*lr_min/(1.-z), min=min_step_lr, max=min_step_lr*10000.).view(-1)
            cont_purification = torch.logical_and(cont_purification, (step_size>config.purification.stopping_alpha))
            if torch.sum(cont_purification)==0:
                break
            step_size *= cont_purification
            x_pur_t = x_pur.clone().detach()
            x_pur = torch.clamp(transform_ebm_to_raw(x_pur_t+grad*step_size[:, None, None, None]), 0.0, 1.0)
            step_sizes.append(step_size)
            images.append(x_pur)
    return images, step_sizes
