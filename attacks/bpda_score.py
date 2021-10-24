import foolbox
import torch
from utils import *
from purification import *

### Classifer PGD attack
# Attack input x
def bpda_score(x, y, network_ebm, network_clf, config):
    if config.structure.clf_log in ["cifar10_wu", "cifar10_carmon", "cifar10_zhang"]: # Temporary
        transform_raw_to_clf = clf_to_raw(raw_to_clf(config.structure.dataset))
    else:
        transform_raw_to_clf = raw_to_clf(config.structure.dataset)
    fmodel = foolbox.PyTorchModel(network_clf, bounds=(0., 1.), preprocessing=foolbox_preprocess(config.structure.dataset))
    x = x.to(config.device.ebm_device)
    y = y.to(config.device.clf_device)
    x_temp = x.clone().detach()
    for i in range(config.attack.iter):
        # get gradient of purified images for n_eot times
        grad = torch.zeros_like(x_temp).to(config.device.ebm_device)
        for j in range(config.attack.n_eot):
            if config.purification.purify_method=="adp_multiple_noise" or config.purification.purify_method=="adp_decision":
                x_temp_eot = adp(x_temp, network_ebm, max_iter=config.purification.max_iter, mode="attack", config=config)[0][-1].to(config.device.clf_device)
            else:
                x_temp_eot = eval(config.purification.purify_method)(x_temp, network_ebm, max_iter=config.purification.max_iter, mode="attack", config=config)[0][-1].to(config.device.clf_device)
            if config.attack.ball_dim==-1:
                attack = foolbox.attacks.LinfPGD(rel_stepsize=0.25, steps=1, random_start=False) # Can be modified for better attack
                _, x_temp_eot_d, _ = attack(fmodel, x_temp_eot, y, epsilons=config.attack.ptb/255.)
            elif config.attack.ball_dim==2:
                attack = foolbox.attacks.L2PGD(rel_stepsize=0.25) # Can be modified for better attack
                _, x_temp_eot_d, _ = attack(fmodel, x_temp_eot, y, epsilons=config.attack.ptb/255.)
            grad += (x_temp_eot_d.detach() - x_temp_eot).to(config.device.ebm_device)
        # Check attack success
        x_clf = transform_raw_to_clf(x_temp.clone().detach()).to(config.device.clf_device)
        success = torch.eq(torch.argmax(network_clf(x_clf), dim=1), y)
        #grad *= success[:, None, None, None] # Attack correctly classified images only
        gradsign = grad.sign()
        labels = torch.ones(x_temp.shape[0], device=x_temp.device)
        labels = labels.long().to(config.device.ebm_device)
        scoresign = network_ebm(x_temp, labels).detach().sign()
        totalsign = (gradsign+scoresign)/2.
        x_temp = torch.clamp(x + torch.clamp(x_temp - x + totalsign*config.attack.alpha/255., -1.0*config.attack.ptb/255., config.attack.ptb/255.), 0.0, 1.0)

    x_adv = x_temp.clone().detach()
    x_clf = transform_raw_to_clf(x_adv.clone().detach()).to(config.device.clf_device)
    success = torch.eq(torch.argmax(network_clf(x_clf), dim=1), y)
    acc = success.float().mean(axis=-1)

    return x_adv, success, acc
