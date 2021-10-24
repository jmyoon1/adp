# Adversarial Purification with Score-based Generative Models
### by [Jongmin Yoon], [Sung Ju Hwang], [Juho Lee]

This repository includes the official PyTorch implementation of our paper:

> Adversarial Purification with Score-based Generative Models
> 
> *Jongmin Yoon, Sung Ju Hwang, Juho Lee*
> 
> the 38th International Conference for Machine Learning (ICML 2021)
> 
> ArXiv: https://arxiv.org/abs/2106.06041

## What does our work do?
We propose a method that gives adversarial robustness to a neural network model against (stochastic) adversarial attacks by using an Energy-based Model (EBM) trained with Denoising Score Matching (DSM), which is called Adversarial denosing purification (ADP).

## Running Codes
### Dependency
Run the following command to install some necessary python packages to run our code.
```
pip install -r requirements.txt
```

### Running code
To run the experiments with `adp.py` or `adp_decision.py`, enter the following command.
```
python main.py --config <config-file>
```
For example, we provide the example configuration file `configs/cifar10_bpda_eot_sigma025_eot15.yml` in the repository.

### Attack and defense
For adversarial attacks, the classifier PGD attack and BPDA+EOT attack are implemented in `attacks/clf_pgd.py` and `attacks/bpda_strong.py`, respectively. At the configuration file, setting the `attack.attack_method` into `clf_pgd` or `bpda_strong` will run these attacks, respectively.
For defense, we implemented the main ADP algorithm and ADP after detecting adversarial examples (Appendix F.) in `purification/adp.py` and `purification/adp_decision.py`, respectively.

### Main components
| File name | Explanation | 
|:-|:-|
| `main.py` | Execute the main code, with initializing configurations and loggers. |
| `runners/empirical.py` | Attacks and purifies the image to show empirical adversarial robustness. |
| `attacks/bpda_strong.py` | Code for BPDA+EOT attack. |
| `purification/adp.py` | Code for adversarial purification. |
| `ncsnv2/*` | Code for training the EBM, i.e., NCSNv2 ([paper](https://arxiv.org/abs/2006.09011), [code](https://github.com/ermongroup/ncsnv2)). |
| `networks/*` | Code for used classifier network architectures. |
| `utils/*` | Utility files. |

### Notes
* For the configuration files, we use the pixel ranges `[0, 255]` for the perturbation scale `attack.ptb` and the one-step attack scale `attack.alpha`. And the main experiments are performed within the pixel range `[0, 1]` after being rescaled during execution.
* For training the EBM and classifier models, we primarily used the pre-existing methods such as NCSNv2 and WideResNet classifier. [Here](https://github.com/meliketoy/wide-resnet.pytorch) is the repository we used for training the WideResNet classifier. Nevertheless, other classifiers, such as the pre-trained adversarially robust classifier implemented in [here](https://robustbench.github.io/) can be used.

## Reference
If you find our work useful for your research, please consider citing this.
```bib
@inproceedings{
yoon2021advpur,
title={Adversarial Purification with Score-based Generative Models},
author={Jongmin Yoon and Sung Ju Hwang and Juho Lee},
booktitle={Proceedings of The 38th International Conference on Machine Learning (ICML 2021)},
year={2021},
}
```

## Contact
For further details, please contact `jm.yoon@kaist.ac.kr`.

## License
MIT

   [Jongmin Yoon]: <http://jmyoon1.github.io>
   [Sung Ju Hwang]: <http://www.sungjuhwang.com>
   [Juho Lee]: <http://juho-lee.github.io>
