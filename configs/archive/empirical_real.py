from utils import *
from clf_models.networks import *
from attacks import *
from datetime import datetime
import torch.optim as optim
import torch.nn.functional as F
import yaml
import functools
import pandas as pd
from ncsn.models.refinenet_dilated_baseline import RefineNetDilated
from robustbench.utils import load_model
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from ncsnv2.runners.ncsn_runner import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

__all__ = ['Empirical']
print = functools.partial(print, flush=True)
class Empirical():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        if not self.config.purification.rand_smoothing:
            self.config.purification.rand_smoothing_level = 0.0
            self.config.purification.rand_smoothing_ensemble = 1
        if self.config.structure.dataset in ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR10-C"]:
            self.nClass = 10
        elif self.config.structure.dataset in ["CIFAR100"]:
            self.nClass = 100
        elif self.config.structure.dataset in ["TinyImageNet"]:
            self.nClass = 200

    def run(self, log_progress):
        # Normalize on classifiers (Non-normalized EBM <-> Normalized CLF especially at large scale)
        transform_raw_to_clf = raw_to_clf(self.config.structure.dataset)

        # Output log file configuration
        sys.stdout = log_progress
        log_output = open(os.path.join(self.args.log, "log_output"), "w")

        # Import dataset
        start_time = datetime.now()
        print("[{}] Start importing dataset {}".format(str(datetime.now()), self.config.structure.dataset))
        if self.config.structure.dataset == 'CIFAR10C':
            testLoader_list, corruption_list = importData(dataset=self.config.structure.dataset, train=False, shuffle=False, bsize=self.config.structure.bsize)
            testLoader = testLoader_list[self.args.CIFARC_CLASS-1][self.args.CIFARC_SEV-1]
        elif self.config.structure.dataset == 'TinyImageNet':
            data_transforms = transforms.Compose([
                    transforms.ToTensor(),
            ])
            data_dir = 'datasets/tiny-imagenet-200'
            image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms)
            testLoader = torch.utils.data.DataLoader(image_datasets, batch_size=100, shuffle=True, num_workers=64)
        else:
            testLoader = importData(dataset=self.config.structure.dataset, train=False, shuffle=False, bsize=self.config.structure.bsize)
        print("[{}] Finished importing dataset {}".format(str(datetime.now()), self.config.structure.dataset))

        # Import classifier networks
        start_time = datetime.now()
        print("[{}] Start importing network".format(str(datetime.now())))
        if self.config.structure.clf_log not in ["cifar10_carmon", "cifar10_wu", "cifar10_zhang"]:
            network_clf = eval(self.config.structure.classifier)().to(self.config.device.clf_device)
            network_clf = torch.nn.DataParallel(network_clf)
        if self.config.structure.dataset in ["MNIST", "FashionMNIST"]:
            states_att = torch.load(os.path.join('clf_models/run/logs', self.config.structure.clf_log, 'checkpoint.pth'), map_location=self.config.device.clf_device)
            optimizer = optim.Adam(network_clf.parameters(), lr=0.001, weight_decay=0., betas=(0.9, 0.999), amsgrad=False)
            network_clf.load_state_dict(states_att[0])
        elif self.config.structure.clf_log in ["cifar10_carmon", "cifar10_wu", "cifar10_zhang"]:
            model_dir = os.path.join("clf_models/run/logs", self.config.structure.clf_log, "models")
            if self.config.structure.clf_log=="cifar10_carmon":
                network_clf = load_model(model_name="Carmon2019Unlabeled", model_dir=model_dir)
            elif self.config.structure.clf_log=="cifar10_wu":
                network_clf = load_model(model_name="Wu2020Adversarial_extra", model_dir=model_dir)
            elif self.config.structure.clf_log=="cifar10_zhang":
                network_clf = load_model(model_name="Zhang2019Theoretically", model_dir=model_dir)
            network_clf = network_clf.to(self.config.device.clf_device)
            optimizer = optim.Adam(network_clf.parameters(), lr=0.001, weight_decay=0., betas=(0.9, 0.999), amsgrad=False)
        elif self.config.structure.dataset in ["CIFAR10", "CIFAR10C", "CIFAR100"]: # CIFAR10 setting, trained by WideResNet
            states_att = torch.load(os.path.join('clf_models/run/logs', self.config.structure.clf_log, 'checkpoint.t7'), map_location=self.config.device.clf_device) # Temporary t7 setting
            optimizer = optim.SGD(network_clf.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
            network_clf = states_att['net'].to(self.config.device.clf_device)
        elif self.config.structure.dataset in ["TinyImageNet"]:
            # Get network_clf from loaded network
            network_clf = torch.load(os.path.join('clf_models/run/logs', self.config.structure.clf_log, 'checkpoint.pth'), map_location=self.config.device.clf_device)
            network_clf = network_clf.to(self.config.device.clf_device)
            optimizer = optim.SGD(network_clf.parameters(), lr=0.001, momentum=0.9)

        # Import purifier networks
        
        with open(os.path.join('ncsnv2/run/logs', self.config.structure.ebm_log, 'config.yml'), 'r') as f:
            config_ebm = yaml.load(f, Loader=yaml.Loader)
            config_ebm.device = self.config.device.ebm_device
        network_ebm = get_model(config_ebm)
        network_ebm = torch.nn.DataParallel(network_ebm)
        states_ebm = torch.load(os.path.join('ncsnv2/run/logs', self.config.structure.ebm_log, 'checkpoint.pth'), map_location=self.config.device.ebm_device)
        network_ebm.load_state_dict(states_ebm[0], strict=True)
        print("[{}] Finished importing networks".format(str(datetime.now())))

        # Initialize dataFrame to save results
        df_series = pd.Series({"pur_ens": self.config.classification.classify_all_steps, \
                               "pur_natural": self.config.purification.purify_natural, \
                               "attack_method": self.config.attack.attack_method, \
                               "start_time": datetime.now(), \
                               "dataset": self.config.structure.dataset, \
                               "log_dir": self.args.log, \
                               "seed": self.args.seed, \
                               "runner": "empirical", \
                               "config": self.config})
        # Candidates: score_norm, cos_sim
        df_columns = ["Epoch", "nData", "att_time", "pur_time", "clf_time", \
                        "std_acc", "att_acc", "pur_acc_l", "pur_acc_s", "pur_acc_o", \
                        "pur_acc_list_l", "pur_acc_list_s", "pur_acc_list_o"]
        if self.config.purification.purify_natural:
            df_columns.append("nat_pur_acc_l")
            df_columns.append("nat_pur_acc_s")
            df_columns.append("nat_pur_acc_o")
            df_columns.append("nat_pur_acc_list_l")
            df_columns.append("nat_pur_acc_list_s")
            df_columns.append("nat_pur_acc_list_o")
        if self.config.classification.classify_all_steps:
            df_columns.append("purens_acc_l") # [1:N] step accuracy
            df_columns.append("purens_acc_s")
            df_columns.append("purens_acc_o")
            df_columns.append("purens_acc_list_l") # [1:N] step accuracy for [1, 2, 3, ..., N] noisy samples
            df_columns.append("purens_acc_list_s")
            df_columns.append("purens_acc_list_o")
            df_columns.append("purens_acc_list_all_l") # [1:1, 1:2, 1:3, ..., 1:N] list
            df_columns.append("purens_acc_list_all_s")
            df_columns.append("purens_acc_list_all_o")
            df_columns.append("purens_acc_list_each_l") # [1, 2, 3, 4, ..., N] list
            df_columns.append("purens_acc_list_each_s")
            df_columns.append("purens_acc_list_each_o")
        if self.config.purification.purify_natural and self.config.classification.classify_all_steps:
            df_columns.append("nat_purens_acc_l")
            df_columns.append("nat_purens_acc_s")
            df_columns.append("nat_purens_acc_o")
            df_columns.append("nat_purens_acc_list_l")
            df_columns.append("nat_purens_acc_list_s")
            df_columns.append("nat_purens_acc_list_o")
            df_columns.append("nat_purens_acc_list_all_l")
            df_columns.append("nat_purens_acc_list_all_s")
            df_columns.append("nat_purens_acc_list_all_o")
            df_columns.append("nat_purens_acc_list_each_l")
            df_columns.append("nat_purens_acc_list_each_s")
            df_columns.append("nat_purens_acc_list_each_o")

        df = pd.DataFrame(columns=df_columns)

        # Run
        for i, (x,y) in enumerate(testLoader):
            if i<self.config.structure.start_epoch or i>self.config.structure.end_epoch:
                continue

            start_time = datetime.now()
            print("[{}] Epoch {}".format(str(datetime.now()), i))
            x = preprocess(x, self.config.structure.dataset)
            x = x.to(self.config.device.ebm_device)
            y = y.to(self.config.device.ebm_device).long()

            ### ATTACK
            x_adv, success, acc = eval(self.config.attack.attack_method)(x, y, network_ebm, network_clf, self.config)
            attack_time = elapsed_seconds(start_time, datetime.now())
            print("[{}] Epoch {}:\t{:.2f} seconds to attack {} data".format(str(datetime.now()), i, attack_time, self.config.structure.bsize))

            ### PURIFICATION
            x_pur_list_list = []
            step_size_list_list = []
            start_time = datetime.now()
            print("[{}] Epoch {}:\tBegin purifying {} attacked images".format(str(datetime.now()), i, self.config.structure.bsize))
            for j in range(self.config.purification.rand_smoothing_ensemble):
                if self.config.purification.purify_method=="adp_multiple_noise":
                    if j<self.config.purification.rand_smoothing_ensemble/2:
                        self.config.purification.rand_smoothing_level = self.config.purification.lowest_level
                    else:
                        self.config.purification.rand_smoothing_level = self.config.purification.highest_level
                    x_pur_list, step_size_list = adp(x_adv, network_ebm, self.config.purification.max_iter, mode="purification", config=self.config)
                else:
                    x_pur_list, step_size_list = eval(self.config.purification.purify_method)(x_adv, network_ebm, self.config.purification.max_iter, mode="purification", config=self.config)
                x_pur_list_list.append(x_pur_list)
                step_size_list_list.append(step_size_list)
                purify_attacked_time = elapsed_seconds(start_time, datetime.now())
            print("[{}] Epoch {}:\t{:.2f} seconds to purify {} attacked images".format(str(datetime.now()), i, purify_attacked_time, self.config.structure.bsize))

            if self.config.purification.purify_natural:
                x_nat_pur_list_list = []
                step_size_nat_list_list = []
                start_time = datetime.now()
                print("[{}] Epoch {}:\tBegin purifying {} natural images".format(str(datetime.now()), i, self.config.structure.bsize))
                for j in range(self.config.purification.rand_smoothing_ensemble):
                    if self.config.purification.purify_method=="adp_multiple_noise":
                        if j<self.config.purification.rand_smoothing_ensemble/2:
                            self.config.purification.rand_smoothing_level = self.config.purification.lowest_level
                        else:
                            self.config.purification.rand_smoothing_level = self.config.purification.highest_level
                        x_nat_pur_list, step_size_nat_list = adp(x, network_ebm, self.config.purification.max_iter, mode="purification", config=self.config)
                    else:
                        x_nat_pur_list, step_size_nat_list = eval(self.config.purification.purify_method)(x, network_ebm, self.config.purification.max_iter, mode="purification", config=self.config)
                    x_nat_pur_list_list.append(x_nat_pur_list)
                    step_size_nat_list_list.append(step_size_list)
                    purify_natural_time = elapsed_seconds(start_time, datetime.now())
                print("[{}] Epoch {}:\t{:.2f} seconds to purify {} natural images".format(str(datetime.now()), i, purify_natural_time, self.config.structure.bsize))

            ### CLASSIFICATION: logit/softmax/onehot
            # Classify natural and attacked images
            with torch.no_grad():
                y_t = network_clf(transform_raw_to_clf(x.clone().detach()).to(self.config.device.clf_device))
                y_adv_t = network_clf(transform_raw_to_clf(x_adv.clone().detach()).to(self.config.device.clf_device))
                nat_correct = torch.eq(torch.argmax(y_t, dim=1), y.clone().to(self.config.device.clf_device)).float().sum()
                att_correct = torch.eq(torch.argmax(y_adv_t, dim=1), y.clone().to(self.config.device.clf_device)).float().sum()

            # Classify all purified attacked images
            with torch.no_grad():
                start_time = datetime.now()
                print("[{}] Epoch {}:\tBegin predicting {} purified attacked images".format(str(datetime.now()), i, self.config.structure.bsize))
                att_list_list_dict = gen_ll(x_pur_list_list, network_clf, transform_raw_to_clf, self.config)
                classify_attacked_time = elapsed_seconds(start_time, datetime.now())
                print("[{}] Epoch {}:\t{:.2f} seconds to predict {} purified attacked images".format(str(datetime.now()), i, classify_attacked_time, self.config.structure.bsize))

                # Classify all purified natural images
                if self.config.purification.purify_natural:
                    start_time = datetime.now()
                    print("[{}] Epoch {}:\tBegin predicting {} purified natural images".format(str(datetime.now()), i, self.config.structure.bsize))
                    nat_list_list_dict = gen_ll(x_nat_pur_list_list, network_clf, transform_raw_to_clf, self.config)
                    classify_natural_time = elapsed_seconds(start_time, datetime.now())
                    print("[{}] Epoch {}:\t{:.2f} seconds to predict {} purified natural images".format(str(datetime.now()), i, classify_natural_time, self.config.structure.bsize))

            ### PERFORMANCE ANALYSIS
            if self.config.classification.classify_all_steps:
                att_acc_purens, att_acc_purens_iter, att_acc_purens_all, att_acc_purens_each = acc_all_step(att_list_list_dict, y, self.config)
            att_acc, att_acc_iter = acc_final_step(att_list_list_dict, y)
            if self.config.purification.purify_natural:
                if self.config.classification.classify_all_steps:
                    nat_acc_purens, nat_acc_purens_iter, nat_acc_purens_all, nat_acc_purens_each = acc_all_step(nat_list_list_dict, y, self.config)
                nat_acc, nat_acc_iter = acc_final_step(nat_list_list_dict, y)

            ### SAVE RESULTS
            new_row = \
                    {
                        "Epoch": i+1,
                        "nData": self.config.structure.bsize,
                        "att_time": attack_time,
                        "pur_time": purify_attacked_time,
                        "clf_time": classify_attacked_time,
                        "std_acc": nat_correct.to('cpu').numpy(),
                        "att_acc": att_correct.to('cpu').numpy(),
                        "pur_acc_l": att_acc["logit"],
                        "pur_acc_s": att_acc["softmax"],
                        "pur_acc_o": att_acc["onehot"],
                        "pur_acc_list_l": att_acc_iter["logit"],
                        "pur_acc_list_s": att_acc_iter["softmax"],
                        "pur_acc_list_o": att_acc_iter["onehot"]
                    }
            if self.config.purification.purify_natural:
                new_row["nat_pur_acc_l"] = nat_acc["logit"]
                new_row["nat_pur_acc_s"] = nat_acc["softmax"]
                new_row["nat_pur_acc_o"] = nat_acc["onehot"]
                new_row["nat_pur_acc_list_l"] = nat_acc_iter["logit"]
                new_row["nat_pur_acc_list_s"] = nat_acc_iter["softmax"]
                new_row["nat_pur_acc_list_o"] = nat_acc_iter["onehot"]
            if self.config.classification.classify_all_steps:
                new_row["purens_acc_l"] = att_acc_purens["logit"]
                new_row["purens_acc_s"] = att_acc_purens["softmax"]
                new_row["purens_acc_o"] = att_acc_purens["onehot"]
                new_row["purens_acc_list_l"] = att_acc_purens_iter["logit"]
                new_row["purens_acc_list_s"] = att_acc_purens_iter["softmax"]
                new_row["purens_acc_list_o"] = att_acc_purens_iter["onehot"]
                new_row["purens_acc_list_all_l"] = att_acc_purens_all["logit"]
                new_row["purens_acc_list_all_s"] = att_acc_purens_all["softmax"]
                new_row["purens_acc_list_all_o"] = att_acc_purens_all["onehot"]
                new_row["purens_acc_list_each_l"] = att_acc_purens_each["logit"]
                new_row["purens_acc_list_each_s"] = att_acc_purens_each["softmax"]
                new_row["purens_acc_list_each_o"] = att_acc_purens_each["onehot"]
            if self.config.purification.purify_natural and self.config.classification.classify_all_steps:
                new_row["nat_purens_acc_l"] = nat_acc_purens["logit"]
                new_row["nat_purens_acc_s"] = nat_acc_purens["softmax"]
                new_row["nat_purens_acc_o"] = nat_acc_purens["onehot"]
                new_row["nat_purens_acc_list_l"] = nat_acc_purens_iter["logit"]
                new_row["nat_purens_acc_list_s"] = nat_acc_purens_iter["softmax"]
                new_row["nat_purens_acc_list_o"] = nat_acc_purens_iter["onehot"]
                new_row["nat_purens_acc_list_all_l"] = nat_acc_purens_all["logit"]
                new_row["nat_purens_acc_list_all_s"] = nat_acc_purens_all["softmax"]
                new_row["nat_purens_acc_list_all_o"] = nat_acc_purens_all["onehot"]
                new_row["nat_purens_acc_list_each_l"] = nat_acc_purens_each["logit"]
                new_row["nat_purens_acc_list_each_s"] = nat_acc_purens_each["softmax"]
                new_row["nat_purens_acc_list_each_o"] = nat_acc_purens_each["onehot"]

            df = df.append(new_row, ignore_index=True)
            df.to_pickle(os.path.join(self.args.log, "df.pkl"))

            ### PLOT
            # (1) Number of denoising samples vs. Accuracy
            acc_vs_denoising_samples(self.args.log)
            if self.config.purification.purify_natural:
                acc_vs_denoising_samples_nat(self.args.log)
            # (2) Number of denoising samples vs. Accuracy, ensemble over all pur steps
            if self.config.classification.classify_all_steps:
                acc_vs_denoising_samples_purens(self.args.log)
                if self.config.purification.purify_natural:
                    acc_vs_denoising_samples_purens_nat(self.args.log)
            # (3) Number of ensembles [1:K] vs. Accuracy
            if self.config.classification.classify_all_steps:
                acc_vs_step_ensemble(self.args.log)
                if self.config.purification.purify_natural:
                    acc_vs_step_ensemble_nat(self.args.log)
