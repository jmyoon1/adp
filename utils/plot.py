import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import yaml

# Accuracy vs. number of denoising samples
def acc_vs_denoising_samples(log_path):
    with open(os.path.join(log_path, "config.yml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    df = pd.read_pickle(os.path.join(log_path, "df.pkl"))
    fig, ax = plt.subplots()
    linsp = np.linspace(1, config.purification.rand_smoothing_ensemble, num=config.purification.rand_smoothing_ensemble)
    correct_l = np.zeros_like(linsp)
    correct_s = np.zeros_like(linsp)
    correct_o = np.zeros_like(linsp)
    nSamples = np.zeros(1)
    for indices, rows in df.iterrows():
        for j in range(config.purification.rand_smoothing_ensemble):
            correct_l[j] += rows["pur_acc_list_l"][j]
            correct_s[j] += rows["pur_acc_list_s"][j]
            correct_o[j] += rows["pur_acc_list_o"][j]
        nSamples += rows["nData"]
    ax.plot(linsp, correct_l/nSamples*100., '-', color="r", label="acc_logit")
    ax.plot(linsp, correct_s/nSamples*100., '-', color="b", label="acc_softmax")
    ax.plot(linsp, correct_o/nSamples*100., '-', color="k", label="acc_onehot")
    plt.legend(loc='upper right', fontsize=18)    
    plt.rc('font', size=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.savefig(os.path.join(log_path, "acc_denoising_iters.pdf"), dpi=800, format="pdf")
    plt.close(fig)

def acc_vs_denoising_samples_nat(log_path):
    with open(os.path.join(log_path, "config.yml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    df = pd.read_pickle(os.path.join(log_path, "df.pkl"))
    fig, ax = plt.subplots()
    linsp = np.linspace(1, config.purification.rand_smoothing_ensemble, num=config.purification.rand_smoothing_ensemble)
    correct_l = np.zeros_like(linsp)
    correct_s = np.zeros_like(linsp)
    correct_o = np.zeros_like(linsp)
    nSamples = np.zeros(1)
    for indices, rows in df.iterrows():
        for j in range(config.purification.rand_smoothing_ensemble):
            correct_l[j] += rows["nat_pur_acc_list_l"][j]
            correct_s[j] += rows["nat_pur_acc_list_s"][j]
            correct_o[j] += rows["nat_pur_acc_list_o"][j]
        nSamples += rows["nData"]
    ax.plot(linsp, correct_l/nSamples*100., '-', color="r", label="acc_logit")
    ax.plot(linsp, correct_s/nSamples*100., '-', color="b", label="acc_softmax")
    ax.plot(linsp, correct_o/nSamples*100., '-', color="k", label="acc_onehot")
    plt.legend(loc='upper right', fontsize=18)    
    plt.rc('font', size=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.savefig(os.path.join(log_path, "nat_acc_denoising_iters.pdf"), dpi=800, format="pdf")
    plt.close(fig)

def acc_vs_denoising_samples_purens(log_path):
    with open(os.path.join(log_path, "config.yml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    df = pd.read_pickle(os.path.join(log_path, "df.pkl"))
    fig, ax = plt.subplots()
    linsp = np.linspace(1, config.purification.rand_smoothing_ensemble, num=config.purification.rand_smoothing_ensemble)
    correct_l = np.zeros_like(linsp)
    correct_s = np.zeros_like(linsp)
    correct_o = np.zeros_like(linsp)
    nSamples = np.zeros(1)
    for indices, rows in df.iterrows():
        for j in range(config.purification.rand_smoothing_ensemble):
            correct_l[j] += rows["purens_acc_list_l"][j]
            correct_s[j] += rows["purens_acc_list_s"][j]
            correct_o[j] += rows["purens_acc_list_o"][j]
        nSamples += rows["nData"]
    ax.plot(linsp, correct_l/nSamples*100., '-', color="r", label="acc_logit")
    ax.plot(linsp, correct_s/nSamples*100., '-', color="b", label="acc_softmax")
    ax.plot(linsp, correct_o/nSamples*100., '-', color="k", label="acc_onehot")
    plt.legend(loc='upper right', fontsize=18)    
    plt.rc('font', size=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.savefig(os.path.join(log_path, "acc_denoising_iters_purens.pdf"), dpi=800, format="pdf")
    plt.close(fig)

def acc_vs_denoising_samples_purens_nat(log_path):
    with open(os.path.join(log_path, "config.yml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    df = pd.read_pickle(os.path.join(log_path, "df.pkl"))
    fig, ax = plt.subplots()
    linsp = np.linspace(1, config.purification.rand_smoothing_ensemble, num=config.purification.rand_smoothing_ensemble)
    correct_l = np.zeros_like(linsp)
    correct_s = np.zeros_like(linsp)
    correct_o = np.zeros_like(linsp)
    nSamples = np.zeros(1)
    for indices, rows in df.iterrows():
        for j in range(config.purification.rand_smoothing_ensemble):
            correct_l[j] += rows["nat_purens_acc_list_l"][j]
            correct_s[j] += rows["nat_purens_acc_list_s"][j]
            correct_o[j] += rows["nat_purens_acc_list_o"][j]
        nSamples += rows["nData"]
    ax.plot(linsp, correct_l/nSamples*100., '-', color="r", label="acc_logit")
    ax.plot(linsp, correct_s/nSamples*100., '-', color="b", label="acc_softmax")
    ax.plot(linsp, correct_o/nSamples*100., '-', color="k", label="acc_onehot")
    plt.legend(loc='upper right', fontsize=18)    
    plt.rc('font', size=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.savefig(os.path.join(log_path, "nat_acc_denoising_iters_purens.pdf"), dpi=800, format="pdf")
    plt.close(fig)

def acc_vs_step_ensemble(log_path):
    with open(os.path.join(log_path, "config.yml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    df = pd.read_pickle(os.path.join(log_path, "df.pkl"))
    fig, ax = plt.subplots()
    linsp = np.linspace(1, config.purification.max_iter, config.purification.max_iter)
    correct_l = np.zeros_like(linsp)
    correct_s = np.zeros_like(linsp)
    correct_o = np.zeros_like(linsp)
    nSamples = np.zeros(1)
    for indices, rows in df.iterrows():
        for j in range(config.purification.max_iter):
            correct_l[j] += rows["purens_acc_list_all_l"][j]
            correct_s[j] += rows["purens_acc_list_all_s"][j]
            correct_o[j] += rows["purens_acc_list_all_o"][j]
    nSamples += rows["nData"]
    ax.plot(linsp, correct_l/nSamples*100., '-', color="r", label="acc_logit")
    ax.plot(linsp, correct_s/nSamples*100., '-', color="b", label="acc_softmax")
    ax.plot(linsp, correct_o/nSamples*100., '-', color="k", label="acc_onehot")
    plt.legend(loc='lower right', fontsize=18)    
    plt.rc('font', size=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.xlabel('purification steps')
    plt.ylabel('accuracy (%)')
    plt.savefig(os.path.join(log_path, "acc_purstepsize.pdf"), dpi=800, format="pdf")
    plt.close(fig)
    
def acc_vs_step_ensemble_nat(log_path):
    with open(os.path.join(log_path, "config.yml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    df = pd.read_pickle(os.path.join(log_path, "df.pkl"))
    fig, ax = plt.subplots()
    linsp = np.linspace(1, config.purification.max_iter, config.purification.max_iter)
    correct_l = np.zeros_like(linsp)
    correct_s = np.zeros_like(linsp)
    correct_o = np.zeros_like(linsp)
    nSamples = np.zeros(1)
    for indices, rows in df.iterrows():
        for j in range(config.purification.max_iter):
            correct_l[j] += rows["nat_purens_acc_list_all_l"][j]
            correct_s[j] += rows["nat_purens_acc_list_all_s"][j]
            correct_o[j] += rows["nat_purens_acc_list_all_o"][j]
        nSamples += rows["nData"]
    ax.plot(linsp, correct_l/nSamples*100., '-', color="r", label="acc_logit")
    ax.plot(linsp, correct_s/nSamples*100., '-', color="b", label="acc_softmax")
    ax.plot(linsp, correct_o/nSamples*100., '-', color="k", label="acc_onehot")
    plt.legend(loc='upper right', fontsize=18)    
    plt.rc('font', size=18)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.savefig(os.path.join(log_path, "nat_acc_purstepsize.pdf"), dpi=800, format="pdf")
    plt.close(fig)