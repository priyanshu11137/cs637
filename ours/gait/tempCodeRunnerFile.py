'''
Command to run:
python check_OOD_drift.py --gpu 0 --cuda --ckpt saved_models/drift.pt --n 20 --save_dir drift_log --transformation_list speed shuffle reverse periodic identity
'''

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
from models.r3d import Regressor as r3d_regressor
from dataset.drift import DriftDataset
import pickle
from distutils.util import strtobool
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--bs', type=int, default=3)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ckpt', default='', help="path load the trained network")
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--trials', type=int, default=1, help='no. of trials for taking average for the final results')
parser.add_argument('--wl', type=int, default=16, help='window length')
parser.add_argument('--n', type=int, default=5, help='number of continuous windows with e-value threshold')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--save_dir', type=str, default='drift_log', help='directory for saving e-values')
parser.add_argument('--cal_root_dir', type=str, default='data/gait-in-neurodegenerative-disease-database-1.0.0',help='calibration data directory')
parser.add_argument('--in_test_root_dir', type=str, default='data/gait-in-neurodegenerative-disease-database-1.0.0',help='test data directory')
parser.add_argument('--out_test_root_dir', type=str, default='data/gait-in-neurodegenerative-disease-database-1.0.0',help='test data directory')
parser.add_argument('--transformation_list', '--names-list', nargs='+', default=["low_pass", "high_pass", "identity"])
parser.add_argument('--disease_type', type=str, default='als', help='als/hunt/park/all')

opt = parser.parse_args()
print(opt)

# Ensure the save directory exists
os.makedirs(opt.save_dir, exist_ok=True)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(opt.gpu) if use_cuda else "cpu")
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

########### Model ##############
net = r3d_regressor(num_classes=len(opt.transformation_list), wl=opt.wl).to(device)
net.load_state_dict(torch.load(opt.ckpt, map_location=device))
net.eval()

criterion = nn.CrossEntropyLoss()

# Function to calculate e-values
def calc_e_value(test_ce_loss, cal_set_ce_loss):
    cal_set_ce_loss_reshaped = cal_set_ce_loss.reshape(1, -1)
    test_ce_loss_reshaped = test_ce_loss.reshape(-1, 1)
    compare = (test_ce_loss_reshaped >= cal_set_ce_loss_reshaped)
    e_value = np.sum(compare, axis=1)
    e_value = (e_value + 1) / (len(cal_set_ce_loss) + 1)
    return e_value

# Function for Fisher combination of e-values using product
def calc_fisher_value_e(e_values):
    return np.prod(e_values)

def calc_fisher_batch_e(e_values, eval_n):
    output = [[None] * len(window) for window in e_values[0]]
    for i in range(len(e_values[0])):
        for j in range(len(e_values[0][i])):
            product = np.prod([e_values[k][i][j][0] for k in range(eval_n)])
            output[i][j] = calc_fisher_value_e(product)
    return output

# Main checkOOD function
def checkOOD(n=opt.n):
    # CAL set CE Loss calculation
    cal_dataset = DriftDataset(root_dir=opt.cal_root_dir, win_len=opt.wl, train=False, cal=True, in_dist_test=False,
                               transformation_list=opt.transformation_list)
    cal_dataloader = DataLoader(cal_dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.workers)
    cal_set_ce_loss_all_iter = calc_cal_ce_loss(opt, model=net, criterion=criterion, device=device, cal_dataloader=cal_dataloader)

    in_test_dataset = DriftDataset(root_dir=opt.in_test_root_dir, win_len=opt.wl, train=False, cal=False, in_dist_test=True,
                                   transformation_list=opt.transformation_list)
    print("In test dataset len: ", in_test_dataset.__len__())
    in_test_ce_loss_all_iters = []
    print("Calculating CE loss for in-dist test data n times")
    for iter in range(opt.n):
        in_test_ce_loss = calc_test_ce_loss(opt, model=net, criterion=criterion, device=device, test_dataset=in_test_dataset)
        in_test_ce_loss_all_iters.append(in_test_ce_loss)
    in_test_ce_loss_all_iters = np.array(in_test_ce_loss_all_iters)

    out_test_dataset = DriftDataset(root_dir=opt.out_test_root_dir, win_len=opt.wl, train=False, cal=False, in_dist_test=False,
                                    transformation_list=opt.transformation_list)
    print("Out test dataset len: ", out_test_dataset.__len__())
    out_test_ce_loss_all_iters = []
    print("Calculating CE For OOD test data n times")
    for iter in range(opt.n):
        out_test_ce_loss = calc_test_ce_loss(opt, model=net, criterion=criterion, device=device, test_dataset=out_test_dataset, in_dist=False)
        out_test_ce_loss_all_iters.append(out_test_ce_loss)
    out_test_ce_loss_all_iters = np.array(out_test_ce_loss_all_iters)

    # Saving CE losses
    np.savez(f"{opt.save_dir}/in_ce_loss_{opt.n}_iters.npz", in_ce_loss=in_test_ce_loss_all_iters)
    np.savez(f"{opt.save_dir}/out_ce_loss_{opt.n}_iters.npz", out_ce_loss=out_test_ce_loss_all_iters)
    np.savez(f"{opt.save_dir}/cal_ce_loss_{opt.n}_iters.npz", ce_loss=cal_set_ce_loss_all_iter)

    print("Calculating n e-values for in-dist test data")
    for iter in range(opt.n):
        in_e_values_all_traces = []
        in_test_ce_loss = in_test_ce_loss_all_iters[iter]
        for test_idx in range(len(in_test_ce_loss)):
            in_e_values = [calc_e_value(in_test_ce_loss[test_idx][window_idx], cal_set_ce_loss_all_iter[iter])
                           for window_idx in range(len(in_test_ce_loss[test_idx]))]
            in_e_values_all_traces.append(np.array(in_e_values))
        np.savez(f"{opt.save_dir}/in_e_values_iter{iter+1}.npz", e_values=np.array(in_e_values_all_traces))

    print("Calculating n e-values for OOD test data")
    for iter in range(opt.n):
        out_e_values_all_traces = []
        out_test_ce_loss = out_test_ce_loss_all_iters[iter]
        for test_idx in range(len(out_test_ce_loss)):
            out_e_values = [calc_e_value(out_test_ce_loss[test_idx][window_idx], cal_set_ce_loss_all_iter[iter])
                            for window_idx in range(len(out_test_ce_loss[test_idx]))]
            out_e_values_all_traces.append(np.array(out_e_values))
        np.savez(f"{opt.save_dir}/out_e_values_iter{iter+1}.npz", e_values=np.array(out_e_values_all_traces))

def eval_detection_fisher(eval_n):
    in_e, out_e = [], []
    for iter in range(eval_n):
        in_e.append(np.load(f"{opt.save_dir}/in_e_values_iter{iter+1}.npz", allow_pickle=True)['e_values'])
        out_e.append(np.load(f"{opt.save_dir}/out_e_values_iter{iter+1}.npz", allow_pickle=True)['e_values'])

    in_fisher_values = calc_fisher_batch_e(in_e, eval_n)
    out_fisher_values = calc_fisher_batch_e(out_e, eval_n)

    in_fisher_per_win = [in_fisher_values[trace_idx][win_idx] for trace_idx in range(len(in_fisher_values))
                         for win_idx in range(len(in_fisher_values[trace_idx]))]
    out_fisher_per_win = [out_fisher_values[trace_idx][win_idx] for trace_idx in range(len(out_fisher_values))
                          for win_idx in range(len(out_fisher_values[trace_idx]))]

    np.savez(f"{opt.save_dir}/in_fisher_iter{eval_n}.npz", in_fisher_values_win=np.array(in_fisher_per_win))
    np.savez(f"{opt.save_dir}/out_fisher_iter{eval_n}.npz", out_fisher_values_win=np.array(out_fisher_per_win))
    return np.array(in_fisher_per_win), np.array(out_fisher_per_win)

def getAUROC(in_fisher_values, out_fisher_values):
    fisher_values = np.concatenate((in_fisher_values, out_fisher_values))
    label = np.concatenate((np.ones(len(in_fisher_values)), np.zeros(len(out_fisher_values))))
    return roc_auc_score(label, fisher_values) * 100

if __name__ == "__main__":
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    auroc_all_trials = []
    for trial in range(opt.trials):
        auroc_one_trial = []
        checkOOD()
        for i in range(opt.n):
            print("Calculating fisher-values for n: ", i+1)
            in_fisher_values_per_win, out_fisher_values_per_win = eval_detection_fisher(i+1)
            au_roc = getAUROC(in_fisher_values_per_win, out_fisher_values_per_win)
            auroc_one_trial.append(au_roc)
        auroc_all_trials.append(auroc_one_trial)

    auroc_all_trials = np.array(auroc_all_trials)
    print("AUROC for CODiT(n=20) on Drift dataset with w=16: ", np.mean(auroc_all_trials, 0)[-1])
