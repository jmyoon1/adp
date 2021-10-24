### packages
# software packages
import torch
import numpy as np
import traceback
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from PIL import Image
import yaml
import logging
import time
import sys
import shutil
from sklearn.manifold import TSNE
from scipy.spatial import Voronoi, voronoi_plot_2d

from runners import *

### main.py
# (1) Import configs (replace argparse. Too many argparse flags now)
def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # Dataset and save logs
    parser.add_argument('--log', default='imgs', help='Output path, including images and logs')
    parser.add_argument('--config', type=str, default='default.yml',  help='Path for saving running related data.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')

    parser.add_argument('--exp_mode', type=str, default='Full', help='Available: [Full, Partial, One]')
    parser.add_argument('--runner', type=str, default='Empirical', help='Available: [Empirical, Certified, Deploy]')

    # Arguments not to be touched
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--CIFARC_CLASS', type=int, default=-1)
    parser.add_argument('--CIFARC_SEV', type=int, default=-1)

    args = parser.parse_args()
    run_id = str(os.getpid())
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    # args.doc = '_'.join([args.doc, run_id, run_time])
    args.log = os.path.join("logs", args.log)

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
        new_config = dict2namespace(config)

    if os.path.exists(args.log):
        shutil.rmtree(args.log)
    os.makedirs(args.log)

    # Save current config to the log directory
    with open(os.path.join(args.log, 'config.yml'), 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)
    
    # setup logger
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main():
    args, config = parse_args_and_config()
    log_progress = open(os.path.join(args.log, "log_progress"), "w")
    sys.stdout = log_progress
    logging.info("Config =")
    print(">" * 80)
    print(config)
    print("<" * 80)

    try:
        runner = eval(args.runner)(args, config)
        runner.run(log_progress)
    except:
        logging.error(traceback.format_exc())

    log_progress.close()

    return 0

if __name__ == '__main__':
    sys.exit(main())