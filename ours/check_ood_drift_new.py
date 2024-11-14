# Modified to use e-values instead of p-values for OOD detection

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import itertools

import numpy as np

from models.r3d import Regressor as r3d_regressor
from dataset.drift import DriftDataset

import pickle
from distutils.util import strtobool
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--bs', type=int, default=2)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ckpt', default='saved_models/drift.pt', help="path to trained network")
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--trials', type=int, default=1, help='no. of trials for averaging final results')
parser.add_argument('--model', type=str, default='r3d', help='c3d/r3d/r21d')
parser.add_argument('--cl', type=int, default=16, help='clip length')
parser.add_argument('--img_size', type=int, default=224, help='img height/width')
parser.add_argument('--n', type=int, default=5, help='number of continuous windows with e-value threshold to detect OODness')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--save_dir', type=str, default='drift_log', help='directory for saving e-values')
parser.add_argument('--cal_root_dir', type=str, default='data/drift_dataset/testing/calibration', help='calibration data directory')
parser.add_argument('--in_test_root_dir', type=str, default='data/drift_dataset/testing/in', help='in-distribution test data directory')
parser.add_argument('--out_test_root_dir', type=str, default='data/drift_dataset/testing/out', help='OOD test data directory')
parser.add_argument('--img_hgt', type=int, default=224, help='img height')
parser.add_argument('--img_width', type=int, default=224, help='img width')
parser.add_argument('--dataset', default='DriftDataset', help='Dataset name')
parser.add_argument("--use_image", type=lambda x:bool(strtobool(x)), default=False, help="Use image info")
parser.add_argument("--use_of", type=lambda x:bool(strtobool(x)), default=True, help="Use optical flow info")
parser.add_argument('--transformation_list', '--names-list', nargs='+', default=["speed","shuffle","reverse","periodic","identity"])

opt = parser.parse_args()
print(opt)

dataset_class = {'DriftDataset': DriftDataset}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(opt.gpu) if use_cuda else "cpu")

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

########### Model Setup ##############
in_channels = 3
if opt.use_image and opt.use_of:
    in_channels = 6
net = r3d_regressor(num_classes=len(opt.transformation_list), in_channels=in_channels).to(device)
net.load_state_dict(torch.load(opt.ckpt, map_location=device))
net.eval()

transforms = transforms.Compose([
            transforms.ToTensor()
        ])

criterion = nn.CrossEntropyLoss()

# Modified e-value calculation function
def calc_e_value(test_ce_loss, cal_set_ce_loss):
    cal_set_ce_loss_reshaped = cal_set_ce_loss.reshape(1, -1)
    test_ce_loss_reshaped = test_ce_loss.reshape(-1, 1)
    
    # Calculate e-value as the likelihood of observing a higher or equal loss in calibration set
    compare = (test_ce_loss_reshaped >= cal_set_ce_loss_reshaped)
    e_value = np.sum(compare, axis=1)
    e_value = (e_value + 1) / (len(cal_set_ce_loss) + 1)
    
    return e_value

# Modify Fisher combination for e-values
def calc_fisher_value_e(t_value, eval_n):
    product = np.prod(t_value[:eval_n])
    return product

# Modified function for batch Fisher calculation with e-values
def calc_fisher_batch_e(e_values, eval_n):
    output = [[None]*len(window) for window in e_values[0]]
    for i in range(len(e_values[0])): 
        for j in range(len(e_values[0][i])): 
            prod = 1
            for k in range(eval_n):
                prod *= e_values[k][i][j][0]

            output[i][j] = calc_fisher_value_e(prod, eval_n)

    return output

# Function to evaluate AUROC
def getAUROC(in_fisher_values, out_fisher_values):
    fisher_values = np.concatenate((in_fisher_values, out_fisher_values))

    indist_label = np.ones(len(in_fisher_values))
    ood_label = np.zeros(len(out_fisher_values))
    label = np.concatenate((indist_label, ood_label))

    au_roc = roc_auc_score(label, fisher_values) * 100
    return au_roc

# Function to calculate TNR at 95% TPR
def getTNR(in_fisher_values, out_fisher_values):
    in_fisher = np.sort(in_fisher_values)[::-1]
    tau = in_fisher[int(0.95 * len(in_fisher))]
    tnr = 100 * (len(out_fisher_values[out_fisher_values < tau]) / len(out_fisher_values))

    return tnr

if __name__ == "__main__":
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    auroc_all_trials = []
    tnr_all_trials = []
    for trial in range(opt.trials):
        auroc_one_trial = []
        tnr_one_trial = []
        # checkOOD() # Uncomment to calculate e-values and Fisher values from scratch
        for i in range(opt.n):
            print("Calculating results for n: {} from the saved e-values".format(i + 1))
            in_fisher_values_per_win, out_fisher_values_per_win = eval_detection_fisher(i + 1)
            au_roc = getAUROC(in_fisher_values_per_win, out_fisher_values_per_win)
            auroc_one_trial.append(au_roc)
            tnr = getTNR(in_fisher_values_per_win, out_fisher_values_per_win)
            tnr_one_trial.append(tnr)
        auroc_all_trials.append(auroc_one_trial)
        tnr_all_trials.append(tnr_one_trial)

    auroc_all_trials = np.array(auroc_all_trials)
    tnr_all_trials = np.array(tnr_all_trials)

    print("AUROC for CODiT(n=20) on Drift dataset with w=16: ", np.mean(auroc_all_trials, 0)[-1])
    print("TNR for CODiT(n=20) on Drift dataset with w=16: ", np.mean(tnr_all_trials, 0)[-1])

