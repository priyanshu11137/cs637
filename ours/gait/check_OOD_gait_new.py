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
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--bs', type=int, default=2)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ckpt', default='saved_models/drift.pt', help="path to trained network")
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--trials', type=int, default=1, help='number of trials for averaging results')
parser.add_argument('--model', type=str, default='r3d', help='c3d/r3d/r21d')
parser.add_argument('--cl', type=int, default=16, help='clip length')
parser.add_argument('--img_size', type=int, default=224, help='image height/width')
parser.add_argument('--n', type=int, default=5, help='number of continuous windows with e-value threshold for OOD detection')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--save_dir', type=str, default='drift_log', help='directory for saving e-values')
parser.add_argument('--cal_root_dir', type=str, default='data/drift_dataset/testing/calibration', help='calibration data directory')
parser.add_argument('--in_test_root_dir', type=str, default='data/drift_dataset/testing/in', help='test data directory')
parser.add_argument('--out_test_root_dir', type=str, default='data/drift_dataset/testing/out', help='test data directory')
parser.add_argument('--img_hgt', type=int, default=224, help='img height')
parser.add_argument('--img_width', type=int, default=224, help='img width')
parser.add_argument('--dataset', default='DriftDataset', help='DriftDataset')
parser.add_argument("--use_image", type=lambda x: bool(strtobool(x)), default=False, help="Use image info")
parser.add_argument("--use_of", type=lambda x: bool(strtobool(x)), default=True, help="use optical flow info")
parser.add_argument('--transformation_list', '--names-list', nargs='+', default=["speed", "shuffle", "reverse", "periodic", "identity"])

opt = parser.parse_args()
print(opt)

# Ensure the save directory exists
os.makedirs(opt.save_dir, exist_ok=True)

dataset_class = {'DriftDataset': DriftDataset}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(opt.gpu) if use_cuda else "cpu")

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

########### model ##############
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

# Function to calculate test loss
def calc_test_ce_loss(opt, model, criterion, device, test_dataset, in_dist=True):
    torch.set_grad_enabled(False)
    model.eval()
    all_traces_ce_loss = []
    transform_losses_dict = {str(i): [] for i in range(5)}

    for test_data_idx in range(test_dataset.__len__()):
        trace_ce_loss = []
        for orig_clip, transformed_clip, transformation in test_dataset.__get_test_item__(test_data_idx):
            orig_clip, transformed_clip = orig_clip.unsqueeze(0).to(device), transformed_clip.unsqueeze(0).to(device)
            target_transformation = torch.tensor([transformation]).to(device)
            output = model(orig_clip, transformed_clip)
            loss = criterion(output, target_transformation)
            transform_losses_dict[str(target_transformation.item())].append(float(loss))
            trace_ce_loss.append(float(loss))
        all_traces_ce_loss.append(np.array(trace_ce_loss))

    save_path = 'in_dist' if in_dist else 'out_dist'
    with open(f'{opt.save_dir}/{save_path}_transform_losses.pickle', 'wb') as handle:
        pickle.dump(transform_losses_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return np.array(all_traces_ce_loss)

# Function to calculate calibration loss
def calc_cal_ce_loss(opt, model, criterion, device, cal_dataloader):
    torch.set_grad_enabled(False)
    model.eval()
    ce_loss_all_iter = []
    transform_losses_dict = {str(i): [] for i in range(5)}

    for iter in range(opt.n):
        ce_loss = []
        for orig_clips, transformed_clips, transformation in cal_dataloader:
            orig_clips, transformed_clips = orig_clips.to(device), transformed_clips.to(device)
            target_transformations = torch.tensor(transformation).to(device)
            outputs = model(orig_clips, transformed_clips)
            for i in range(len(outputs)):
                loss = criterion(outputs[i].unsqueeze(0), target_transformations[i].unsqueeze(0))
                ce_loss.append(loss.item())
                transform_losses_dict[str(target_transformations[i].item())].append(float(loss))
        ce_loss_all_iter.append(np.array(ce_loss))

    with open(f'{opt.save_dir}/cal_transform_losses.pickle', 'wb') as handle:
        pickle.dump(transform_losses_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return np.array(ce_loss_all_iter)

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
    cal_dataset = dataset_class[opt.dataset](root_dir=opt.cal_root_dir, clip_len=opt.cl, train=False, cal=True,
                                             transforms_=transforms, img_hgt=opt.img_hgt, img_width=opt.img_width,
                                             use_image=opt.use_image, use_of=opt.use_of,
                                             transformation_list=opt.transformation_list)
    cal_dataloader = DataLoader(cal_dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.workers)

    cal_set_ce_loss_all_iter = calc_cal_ce_loss(opt, model=net, criterion=criterion, device=device,
                                                cal_dataloader=cal_dataloader)

    in_test_dataset = dataset_class[opt.dataset](root_dir=opt.in_test_root_dir, clip_len=opt.cl, train=False, cal=False,
                                                 transforms_=transforms, img_hgt=opt.img_hgt, img_width=opt.img_width,
                                                 in_dist_test=True, use_image=opt.use_image, use_of=opt.use_of,
                                                 transformation_list=opt.transformation_list)

    print("In test dataset len: ", in_test_dataset.__len__())
    in_test_ce_loss_all_iters = []
    print("Calculating CE loss for in-dist test data n times")
    for iter in range(opt.n):
        print('iter: ',iter+1)
        in_test_ce_loss = calc_test_ce_loss(opt, model=net, criterion=criterion, device=device,
                                            test_dataset=in_test_dataset)
        in_test_ce_loss_all_iters.append(in_test_ce_loss)
    in_test_ce_loss_all_iters = np.array(in_test_ce_loss_all_iters)

    out_test_dataset = dataset_class[opt.dataset](root_dir=opt.out_test_root_dir, clip_len=opt.cl, train=False,
                                                  cal=False, transforms_=transforms, img_hgt=opt.img_hgt,
                                                  img_width=opt.img_width, in_dist_test=False,
                                                  use_image=opt.use_image, use_of=opt.use_of,
                                                  transformation_list=opt.transformation_list)
    print("Out test dataset len: ", out_test_dataset.__len__())
    out_test_ce_loss_all_iters = []
    print("Calculating CE For OOD test data n times")
    for iter in range(opt.n):
        out_test_ce_loss = calc_test_ce_loss(opt, model=net, criterion=criterion, device=device,
                                             test_dataset=out_test_dataset, in_dist=False)
        out_test_ce_loss_all_iters.append(out_test_ce_loss)
    out_test_ce_loss_all_iters = np.array(out_test_ce_loss_all_iters)

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

    np.savez(f"{opt.save_dir}/in_fisher_iter{iter+1}.npz", in_fisher_values_win=np.array(in_fisher_per_win))
    np.savez(f"{opt.save_dir}/out_fisher_iter{iter+1}.npz", out_fisher_values_win=np.array(out_fisher_per_win))
    return np.array(in_fisher_per_win), np.array(out_fisher_per_win)

def getAUROC(in_fisher_values, out_fisher_values):
    fisher_values = np.concatenate((in_fisher_values, out_fisher_values))
    label = np.concatenate((np.ones(len(in_fisher_values)), np.zeros(len(out_fisher_values))))
    return roc_auc_score(label, fisher_values) * 100

def getTNR(in_fisher_values, out_fisher_values):
    tau = np.sort(in_fisher_values)[::-1][int(0.95 * len(in_fisher_values))]
    return 100 * (len(out_fisher_values[out_fisher_values < tau]) / len(out_fisher_values))

if __name__ == "__main__":
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    auroc_all_trials, tnr_all_trials = [], []
    for trial in range(opt.trials):
        auroc_one_trial, tnr_one_trial = [], []
        checkOOD()
        for i in range(opt.n):
            print("Calculating results for n: {} from the saved fisher-values".format(i+1))
            in_fisher_per_win, out_fisher_per_win = eval_detection_fisher(i + 1)
            auroc_one_trial.append(getAUROC(in_fisher_per_win, out_fisher_per_win))
            tnr_one_trial.append(getTNR(in_fisher_per_win, out_fisher_per_win))
        auroc_all_trials.append(auroc_one_trial)
        tnr_all_trials.append(tnr_one_trial)

    auroc_all_trials, tnr_all_trials = np.array(auroc_all_trials), np.array(tnr_all_trials)
    print("AUROC for CODiT(n=20) on Drift dataset with w=16: ", np.mean(auroc_all_trials, 0)[-1])
    print("TNR for CODiT(n=20) on Drift dataset with w=16: ", np.mean(tnr_all_trials, 0)[-1])
