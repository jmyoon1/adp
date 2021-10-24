# ebm_to_clf: In some classifiers, we use batch normalized input
import torchvision.transforms  as transforms

class transform_grid_to_raw(object):
    def __call__(self, tensor):
        tensor *= 256./255.
        tensor -= 1./510.
        return tensor

def ebm_to_clf(dset):
    if dset in ["CIFAR10", "CIFAR10C"]:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose(
            [
                transform_grid_to_raw(),
                transforms.Normalize(mean, std)
            ]
        )
        return transform
    elif dset in ["CIFAR100"]:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        transform = transforms.Compose(
            [
                transform_grid_to_raw(),
                transforms.Normalize(mean, std)
            ]
        )
        return transform
    elif dset in ["TinyImageNet"]:
        mean = (a,b,c)
        std = (d,e,f)
        transform = transforms.Compose(
            [
                transform_grid_to_raw(),
                transforms.Normalize(mean, std)
            ]
        )
        return transform
    else:
        transform = transforms.Compose(
            [
                transform_grid_to_raw()
            ]
        )
        return transform
