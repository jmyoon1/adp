import foolbox
import torch
from utils import *

### Classifer PGD attack
# Attack input x
def clf_pgd(x, y, network_ebm, network_clf, config):
    x = x.to(config.device.clf_device)
    y = y.to(config.device.clf_device)
    fmodel = foolbox.PyTorchModel(network_clf, bounds=(0., 1.), preprocessing=foolbox_preprocess(config.structure.dataset))
    if config.attack.ball_dim==-1:
        attack = foolbox.attacks.LinfPGD(rel_stepsize=0.25) # Can be modified for better attack
        _, x_adv, success = attack(fmodel, x, y, epsilons=config.attack.ptb/256.)
        acc = 1 - success.float().mean(axis=-1)
    elif config.attack.ball_dim==2:
        attack = foolbox.attacks.L2PGD(rel_stepsize=0.25) # Can be modified for better attack
        _, x_adv, success = attack(fmodel, x, y, epsilons=config.attack.ptb/256.)
        acc = 1 - success.float().mean(axis=-1)
    return x_adv, success, acc