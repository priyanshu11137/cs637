'''
Command to run:
python check_OOD_gait.py --save_dir gait_log/ --ckpt saved_models/gait_16.pt  --transformation_list high_pass low_high high_low identity --wl 16 --cuda --gpu 0 --n 100 --disease_type als
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
from models.lenet import Regressor as regressor
from dataset.gait import GAIT
from distutils.util import strtobool
from sklearn.metrics import roc_auc_score
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--bs', type=int, default=3)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ckpt', default='', help="path to load the trained network")
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--trials', type=int, default=1, help='no. of trials for taking average for the final results')
parser.add_argument('--wl', type=int, default=16, help='window length')
parser.add_argument('--n', type=int, default=100, help='number of continuous windows for OOD detection')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--save_dir', type=str, default='gait_log', help='directory for saving e-values')
parser.add_argument('--cal_root_dir', type=str, default='data/gait-in-neurodegenerative-disease-database-1.0.0', help='calibration data directory')
parser.add_argument('--in_test_root_dir', type=str, default='data/gait-in-neurodegenerative-disease-database-1.0.0', help='in-distribution test data directory')
parser.add_argument('--out_test_root_dir', type=str, default='data/gait-in-neurodegenerative-disease-database-1.0.0', help='out-of-distribution test data directory')
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

########### model ##############
net = regressor(num_classes=len(opt.transformation_list), wl=opt.wl).to(device)
net.load_state_dict(torch.load(opt.ckpt, map_location=device))
net.eval()

criterion = nn.CrossEntropyLoss()

def calc_test_ce_loss(opt, model, criterion, device, test_dataset, in_dist=True):
    torch.set_grad_enabled(False)
    model.eval()
    all_traces_ce_loss = []
    transform_losses_dict = {str(i): [] for i in range(len(opt.transformation_list))}

    for test_data_idx in range(test_dataset.__len__()):
        trace_ce_loss = []
        for orig_window, transformed_window, transformation in test_dataset.__get_test_item__(test_data_idx):
            orig_window, transformed_window = orig_window.unsqueeze(0).to(device), transformed_window.unsqueeze(0).to(device)
            target_transformation = torch.tensor([transformation]).to(device)
            output = model(orig_window, transformed_window)
            loss = criterion(output, target_transformation)
            transform_losses_dict[str(target_transformation.item())].append(float(loss))
            trace_ce_loss.append(float(loss))
        all_traces_ce_loss.append(np.array(trace_ce_loss))

    save_path = 'in_dist' if in_dist else 'out_dist'
    with open(f'{opt.save_dir}/{save_path}_transform_losses.pickle', 'wb') as handle:
        pickle.dump(transform_losses_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return all_traces_ce_loss


def calc_cal_ce_loss(opt, model, criterion, device, cal_dataloader):
    torch.set_grad_enabled(False)
    model.eval()
    ce_loss_all_iter = []
    transform_losses_dict = {str(i): [] for i in range(len(opt.transformation_list))}

    for iter in range(opt.n):
        ce_loss = []
        for orig_windows, transformed_windows, transformation in cal_dataloader:
            orig_windows, transformed_windows = orig_windows.to(device), transformed_windows.to(device)
            target_transformations = torch.tensor(transformation).to(device)
            outputs = model(orig_windows, transformed_windows)
            for i in range(len(outputs)):
                loss = criterion(outputs[i].unsqueeze(0), target_transformations[i].unsqueeze(0))
                ce_loss.append(loss.item())
                transform_losses_dict[str(target_transformations[i].item())].append(float(loss))
        ce_loss_all_iter.append(np.array(ce_loss))

    with open(f'{opt.save_dir}/cal_transform_losses.pickle', 'wb') as handle:
        pickle.dump(transform_losses_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return np.array(ce_loss_all_iter)

def calc_e_value(test_ce_loss, cal_set_ce_loss):
    cal_set_ce_loss_reshaped = cal_set_ce_loss.reshape(1, -1)
    test_ce_loss_reshaped = test_ce_loss.reshape(-1, 1)
    compare = (test_ce_loss_reshaped >= cal_set_ce_loss_reshaped)
    e_value = np.sum(compare, axis=1)
    e_value = (e_value + 1) / (len(cal_set_ce_loss) + 1)
    return e_value

def checkOOD(n=opt.n):
    cal_dataset = GAIT(root_dir=opt.cal_root_dir, win_len=opt.wl, train=False, cal=True, in_dist_test=False, transformation_list=opt.transformation_list)
    cal_dataloader = DataLoader(cal_dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.workers)
    cal_set_ce_loss_all_iter = calc_cal_ce_loss(opt, model=net, criterion=criterion, device=device, cal_dataloader=cal_dataloader)

    in_test_dataset = GAIT(root_dir=opt.in_test_root_dir, win_len=opt.wl, train=False, cal=False, in_dist_test=True, transformation_list=opt.transformation_list)
    print("In test dataset len: ", in_test_dataset.__len__())
    in_test_ce_loss_all_iters = []
    for iter in range(opt.n):
        print('iter: ', iter + 1)
        in_test_ce_loss = calc_test_ce_loss(opt, model=net, criterion=criterion, device=device, test_dataset=in_test_dataset)
        in_test_ce_loss_all_iters.append(in_test_ce_loss)
    import numpy as np

    # Find the maximum length of traces across all iterations
    max_trace_len = max(max(len(trace) for trace in iter_losses) for iter_losses in in_test_ce_loss_all_iters)
    
    # Pad each trace in each iteration to the maximum trace length
    in_test_ce_loss_all_iters = [
        [np.pad(trace, (0, max_trace_len - len(trace)), constant_values=np.nan) for trace in iter_losses]
        for iter_losses in in_test_ce_loss_all_iters
    ]
    
    # Convert the padded list to a 3D NumPy array
    in_test_ce_loss_all_iters = np.array(in_test_ce_loss_all_iters)
    out_test_dataset = GAIT(root_dir=opt.out_test_root_dir, win_len=opt.wl, train=False, cal=False, in_dist_test=False, transformation_list=opt.transformation_list, disease_type=opt.disease_type)
    print("Out test dataset len: ", out_test_dataset.__len__())
    out_test_ce_loss_all_iters = []
    for iter in range(opt.n):
        print('iter: ', iter + 1)
        out_test_ce_loss = calc_test_ce_loss(opt, model=net, criterion=criterion, device=device, test_dataset=out_test_dataset, in_dist=False)
        out_test_ce_loss_all_iters.append(out_test_ce_loss)
    import numpy as np
    
    # Find the maximum trace length across all iterations for out-dist test loss
    max_out_trace_len = max(max(len(trace) for trace in iter_losses) for iter_losses in out_test_ce_loss_all_iters)
    # Pad each trace in out_test_ce_loss_all_iters to the maximum trace length
    out_test_ce_loss_all_iters = [
        [np.pad(trace, (0, max_out_trace_len - len(trace)), constant_values=np.nan) for trace in iter_losses]
        for iter_losses in out_test_ce_loss_all_iters
    ]
    # Convert the padded list to a 3D NumPy array
    out_test_ce_loss_all_iters= np.array(out_test_ce_loss_all_iters)

    np.savez(f"{opt.save_dir}/in_ce_loss_{opt.n}_iters.npz", in_ce_loss=in_test_ce_loss_all_iters)
    np.savez(f"{opt.save_dir}/out_ce_loss_{opt.n}_iters.npz", out_ce_loss=out_test_ce_loss_all_iters)
    np.savez(f"{opt.save_dir}/cal_ce_loss_{opt.n}_iters.npz", ce_loss=cal_set_ce_loss_all_iter)

    for iter in range(opt.n):
        in_e_values_all_traces = []
        in_test_ce_loss = in_test_ce_loss_all_iters[iter]
        for test_idx in range(len(in_test_ce_loss)):
            in_e_values = [calc_e_value(in_test_ce_loss[test_idx][window_idx], cal_set_ce_loss_all_iter[iter])
                           for window_idx in range(len(in_test_ce_loss[test_idx]))]
            in_e_values_all_traces.append(np.array(in_e_values))
        np.savez(f"{opt.save_dir}/in_e_values_iter{iter+1}.npz", e_values=np.array(in_e_values_all_traces))

    for iter in range(opt.n):
        out_e_values_all_traces = []
        out_test_ce_loss = out_test_ce_loss_all_iters[iter]
        for test_idx in range(len(out_test_ce_loss)):
            out_e_values = [calc_e_value(out_test_ce_loss[test_idx][window_idx], cal_set_ce_loss_all_iter[iter])
                            for window_idx in range(len(out_test_ce_loss[test_idx]))]
            out_e_values_all_traces.append(np.array(out_e_values))
        np.savez(f"{opt.save_dir}/out_e_values_iter{iter+1}.npz", e_values=np.array(out_e_values_all_traces))

def calc_fisher_value(t_value, eval_n):
    summation = 0
    for i in range(eval_n):
        summation += ((-np.log(t_value)) ** i) / np.math.factorial(i)
    return t_value * summation

def calc_fisher_batch_e(e_values, eval_n):
    output = [[None] * len(window) for window in e_values[0]]
    for i in range(len(e_values[0])):
        for j in range(len(e_values[0][i])):
            prod = np.prod([e_values[k][i][j][0] for k in range(eval_n)])
            output[i][j] = calc_fisher_value(prod, eval_n)
    return output

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

    np.savez(f"{opt.save_dir}/in_fisher_iter{iter+1}.npz", in_fisher_values_win=np.array(in_fisher_per_win))
    np.savez(f"{opt.save_dir}/out_fisher_iter{iter+1}.npz", out_fisher_values_win=np.array(out_fisher_per_win))
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
    print("AUROC for CODiT(n=100) on {} as OOD data with window length {} is {}".format(opt.disease_type, opt.wl, np.mean(auroc_all_trials, 0)[-1]))
