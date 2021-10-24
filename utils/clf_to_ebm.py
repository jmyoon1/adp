# ebm_to_clf: In some classifiers, we use batch normalized input
import torchvision.transforms  as transforms

class transform_raw_to_grid(object):
    def __call__(self,tensor):
        tensor *= 255./256.
        tensor += 1./512.
        return tensor

def clf_to_ebm(dset):
    if dset in ["CIFAR10", "CIFAR10C"]:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose(
            [
                transforms.Normalize((-1.0*mean[0]/std[0], -1.0*mean[1]/std[1], -1.0*mean[2]/std[2]), (1./std[0], 1./std[1], 1./std[2])),
                transform_raw_to_grid()
            ]
        )
        return transform
    elif dset in ["CIFAR100"]:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        transform = transforms.Compose(
            [
                transforms.Normalize((-1.0*mean[0]/std[0], -1.0*mean[1]/std[1], -1.0*mean[2]/std[2]), (1./std[0], 1./std[1], 1./std[2])),
                transform_raw_to_grid()
            ]
        )
        return transform
    elif dset in ["TinyImageNet"]:
        mean = (a,b,c)
        std = (d,e,f)
        transform = transforms.Compose(
            [
                transforms.Normalize((-1.0*mean[0]/std[0], -1.0*mean[1]/std[1], -1.0*mean[2]/std[2]), (1./std[0], 1./std[1], 1./std[2])),
                transform_raw_to_grid()
            ]
        )
        return transform
    else:
        transform = transforms.Compose(
            [
                transform_raw_to_grid()
            ]
        )
        return transform
