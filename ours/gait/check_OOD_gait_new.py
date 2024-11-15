'''
command to run 
python check_OOD_gait.py --save_dir gait_log/ --ckpt saved_models/gait_16.pt  --transformation_list high_pass low_high high_low identity --wl 16 --cuda --gpu 0 --n 100 --disease_type als

'''


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
import torchvision.utils as vutils
from torch.utils.data import DataLoader, random_split
import itertools

import numpy as np

from models.lenet import Regressor as regressor

from dataset.gait import GAIT

import pdb
from distutils.util import strtobool

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--bs', type=int, default=3)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ckpt', default='', help="path load the trained network")
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--trials', type=int, default=1, help='no. of trials for taking average for the final results')
parser.add_argument('--wl', type=int, default=16, help='window length')
parser.add_argument('--n', type=int, default=5, help='number of continuous windows with p-value < epsilon to detect OODness in the trace')
parser.add_argument('--seed', type=int, default=100, help='random seed')
parser.add_argument('--save_dir', type=str, default='win64', help='directory for saving p-vaues')
parser.add_argument('--cal_root_dir', type=str, default='data/gait-in-neurodegenerative-disease-database-1.0.0',help='calibration data directory')
parser.add_argument('--in_test_root_dir', type=str, default='data/gait-in-neurodegenerative-disease-database-1.0.0',help='test data directory')
parser.add_argument('--out_test_root_dir', type=str, default='data/gait-in-neurodegenerative-disease-database-1.0.0',help='test data directory')
parser.add_argument('--transformation_list', '--names-list', nargs='+', default=["low_pass", "high_pass", "identity"])
parser.add_argument('--disease_type', type=str, default='als', help='als/hunt/park/all')


opt = parser.parse_args()
print(opt)
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

# pdb.set_trace()

criterion = nn.CrossEntropyLoss()
import numpy as np

def compute_threshold(test_statistics, null_statistics, level=0.2):
    """
    Compute the threshold based on the combined test statistics (in-dist + OOD)
    and the null (calibration) statistics. The threshold is determined such
    that the false discovery proportion (FDP) is controlled at the given level.
    """
    n, m = len(null_statistics), len(test_statistics)
    mixed_statistics = np.concatenate([null_statistics, test_statistics])
    sample_ind = np.concatenate([np.ones(len(null_statistics)), np.zeros(len(test_statistics))])

    sample_ind_sort = sample_ind[np.argsort(-mixed_statistics)]
    fdp = 1
    V = n
    K = m
    l = m + n

    while fdp > level and K >= 1:
        l -= 1
        if sample_ind_sort[l] == 1:
            V -= 1
        else:
            K -= 1
        fdp = (V * m) / (n * K) if K else 1

    mixed_statistics_sort_ind = np.argsort(-mixed_statistics)
    if fdp > level:
        threshold = mixed_statistics[mixed_statistics_sort_ind[0]] + 1
    else:
        threshold = mixed_statistics[mixed_statistics_sort_ind[l-1]]
    
    return threshold


def compute_evalue(test_statistics, null_statistics, t):
    """
    Compute the e-values based on test statistics and calibration (null) statistics
    with respect to a threshold t.
    """
    denominator = (1 + np.sum(null_statistics >= t)) / (1 + null_statistics.shape[0])
    evalues = ((test_statistics >= t).astype(int) / denominator)
    return evalues


def evalue_to_pvalue(evalues, cal_evalues):
    """
    Convert e-values to p-values using the empirical cumulative distribution
    of calibration e-values.
    """
    p_values = [(np.sum(cal_evalues >= e) / len(cal_evalues)) for e in evalues]
    return np.array(p_values)


def calc_test_ce_loss(opt, model, criterion, device, test_dataset, in_dist=True):
    torch.set_grad_enabled(False)
    model.eval()

    all_traces_ce_loss = []

    key_list = ["0", "1", "2", "3", "4"]
    trasform_losses_dictionary = dict.fromkeys(key_list)
    for key in key_list:
         trasform_losses_dictionary[key] = []

    for test_data_idx in range(0, test_dataset.__len__()): # loop over all test datapoints
        
        trace_ce_loss = []
        
        for orig_window, transformed_window, transformation in test_dataset.__get_test_item__(test_data_idx): # loop over sliding window in the test trace
            
            orig_window = orig_window.unsqueeze(0)
            transformed_window = transformed_window.unsqueeze(0)
            orig_window = orig_window.to(device)
            transformed_window = transformed_window.to(device)
            transformation = [transformation]
            target_transformation = torch.tensor(transformation).to(device)
            # forward
            output = model(orig_window, transformed_window)
            # print("Output: {} and target: {}".format(torch.argmax(output), target_transformation))
            loss = criterion(output, target_transformation)
            # print("Output: {}, target: {}, loss: {}".format(torch.argmax(output), target_transformation, float(loss)))
            trasform_losses_dictionary['{}'.format(target_transformation.item())].append(float(loss))
            # print("Loss: ", float(loss))
            trace_ce_loss.append(float(loss))

        all_traces_ce_loss.append(np.array(trace_ce_loss))
    
    import pickle 
    if in_dist:
        with open('{}/in_dist_transform_losses.pickle'.format(opt.save_dir), 'wb') as handle:
            pickle.dump(trasform_losses_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open('{}/out_dist_transform_losses.pickle'.format(opt.save_dir), 'wb') as handle:
            pickle.dump(trasform_losses_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return all_traces_ce_loss

def calc_cal_ce_loss(opt, model, criterion, device, cal_dataloader): # for calibration datapoint, we want one randomly sampled window for 1 datapoint
    torch.set_grad_enabled(False)
    model.eval()

    ce_loss_all_iter = []

    # torch.manual_seed(opt.seed)
    # np.random.seed(opt.seed)
    # random.seed(opt.seed)

    # definning dictionary for saving losses
    key_list = ["0", "1", "2", "3", "4"]
    trasform_losses_dictionary = dict.fromkeys(key_list)
    for key in key_list:
         trasform_losses_dictionary[key] = []

    print("Calculating CE For calibration data n times")
    for iter in range(0, opt.n): # n iterations with random sampling of windows and transformations on calibration datapoints
        print("n: ", iter+1)
        ce_loss = []
        for _, data in enumerate(cal_dataloader, 1): # iteration over all calibration datapoints
            # get inputs
            orig_windows, transformed_windows, transformation = data
            orig_windows = orig_windows.to(device)
            transformed_windows = transformed_windows.to(device)
            target_transformations = torch.tensor(transformation).to(device)
            # forward
            outputs = model(orig_windows, transformed_windows)
            for i in range(len(outputs)):
                loss = criterion(outputs[i].unsqueeze(0), target_transformations[i].unsqueeze(0))
                ce_loss.append(loss.item())
                # print("Loss: {}, transformation: {}, predicted trans: {}".format(loss.item(), transformation[i], outputs[i]))
                trasform_losses_dictionary['{}'.format(target_transformations[i].item())].append(float(loss))

        #print('[Cal] loss: ', ce_loss)
        ce_loss_all_iter.append(np.array(ce_loss))
    
    import pickle
    with open('{}/cal_transform_losses.pickle'.format(opt.save_dir), 'wb') as handle:
            pickle.dump(trasform_losses_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return ce_loss_all_iter

def calc_p_value(test_ce_loss, cal_set_ce_loss):

    cal_set_ce_loss_reshaped = cal_set_ce_loss
    cal_set_ce_loss_reshaped = cal_set_ce_loss_reshaped.reshape(1,-1) # cal_set_ce_loss reshaped into row vector

    test_ce_loss_reshaped = test_ce_loss
    test_ce_loss_reshaped = test_ce_loss_reshaped.reshape(-1,1) # test_ce_loss reshaped into column vector

    #pdb.set_trace()
    compare = (test_ce_loss_reshaped)<=(cal_set_ce_loss_reshaped)
    p_value = np.sum(compare, axis=1)
    p_value = (p_value+1)/(len(cal_set_ce_loss)+1)
    # print(p_value)

    return p_value


def checkOOD(n=opt.n):
    # Calibration CE Loss
    cal_dataset = GAIT(root_dir=opt.cal_root_dir, win_len=opt.wl, train=False, cal=True, in_dist_test=False, transformation_list=opt.transformation_list)
    print("Cal dataset len:", cal_dataset.__len__())
    cal_dataloader = DataLoader(cal_dataset, batch_size=opt.bs, shuffle=False, num_workers=opt.workers)
    
    cal_set_ce_loss_all_iter = calc_cal_ce_loss(opt, model=net, criterion=criterion, device=device, cal_dataloader=cal_dataloader)
    
    ############################################################################################################
    
    # In-Dist and OOD Test CE Losses
    in_test_dataset = GAIT(root_dir=opt.in_test_root_dir, win_len=opt.wl, train=False, cal=False, in_dist_test=True, transformation_list=opt.transformation_list)
    print("In test dataset len:", in_test_dataset.__len__())
    
    out_test_dataset = GAIT(root_dir=opt.out_test_root_dir, win_len=opt.wl, train=False, cal=False, in_dist_test=False, transformation_list=opt.transformation_list, disease_type=opt.disease_type)
    print("Out test dataset len:", out_test_dataset.__len__())
    
    in_test_ce_loss_all_iters = []
    out_test_ce_loss_all_iters = []
    thresholds = []

    # Calculate CE Loss and Thresholds for each iteration
    print("Calculating CE for OOD and ID test data and computing thresholds")
    for iter in range(n):
        print('Iteration:', iter + 1)

        # Calculate CE Loss for In-Distribution Test Data
        in_test_ce_loss = calc_test_ce_loss(opt, model=net, criterion=criterion, device=device, test_dataset=in_test_dataset)
        in_test_ce_loss_all_iters.append(in_test_ce_loss)

        # Calculate CE Loss for Out-Distribution Test Data
        out_test_ce_loss = calc_test_ce_loss(opt, model=net, criterion=criterion, device=device, test_dataset=out_test_dataset, in_dist=False)
        out_test_ce_loss_all_iters.append(out_test_ce_loss)

        # Combine In-Distribution and Out-Distribution Test Losses
        combined_test_losses = np.concatenate((in_test_ce_loss, out_test_ce_loss), axis=0)

        # Calculate Threshold for E-value Computation
        threshold = compute_threshold(combined_test_losses, cal_set_ce_loss_all_iter[iter], level=0.2)
        thresholds.append(threshold)
    
    # Pad CE Loss Arrays
    max_trace_len_in = max(max(len(trace) for trace in iter_losses) for iter_losses in in_test_ce_loss_all_iters)
    in_test_ce_loss_all_iters = [
        [np.pad(trace, (0, max_trace_len_in - len(trace)), constant_values=np.nan) for trace in iter_losses]
        for iter_losses in in_test_ce_loss_all_iters
    ]
    in_test_ce_loss_all_iters = np.array(in_test_ce_loss_all_iters)

    max_trace_len_out = max(max(len(trace) for trace in iter_losses) for iter_losses in out_test_ce_loss_all_iters)
    out_test_ce_loss_all_iters = [
        [np.pad(trace, (0, max_trace_len_out - len(trace)), constant_values=np.nan) for trace in iter_losses]
        for iter_losses in out_test_ce_loss_all_iters
    ]
    out_test_ce_loss_all_iters = np.array(out_test_ce_loss_all_iters)

    # Save CE Losses
    np.savez(f"{opt.save_dir}/in_ce_loss_{n}_iters.npz", in_ce_loss=in_test_ce_loss_all_iters)
    np.savez(f"{opt.save_dir}/out_ce_loss_{n}_iters.npz", out_ce_loss=out_test_ce_loss_all_iters)
    np.savez(f"{opt.save_dir}/cal_ce_loss_{n}_iters.npz", ce_loss=cal_set_ce_loss_all_iter)

    ############################################################################################################
    
    # E-value Calculation and P-value Conversion
    for iter in range(n):
        # In-Distribution E-values and P-values
        in_evalues_all_traces = []
        in_p_values_all_traces = []
        cal_evalues = compute_evalue(np.array(cal_set_ce_loss_all_iter[iter]), cal_set_ce_loss_all_iter[iter], thresholds[iter])

        for test_idx in range(len(in_test_ce_loss_all_iters[iter])):
            in_evalues = compute_evalue(np.array(in_test_ce_loss_all_iters[iter][test_idx]), cal_set_ce_loss_all_iter[iter], thresholds[iter])
            in_evalues_all_traces.append(in_evalues)

            # Convert E-values to P-values
            in_p_values = evalue_to_pvalue(in_evalues, cal_evalues)
            in_p_values_all_traces.append(in_p_values)

        # Save In-Distribution P-values
        np.savez(f"{opt.save_dir}/in_p_values_iter{iter+1}.npz", p_values=np.array(in_p_values_all_traces))

        ########################################################################################################

        # Out-Distribution E-values and P-values
        out_evalues_all_traces = []
        out_p_values_all_traces = []

        for test_idx in range(len(out_test_ce_loss_all_iters[iter])):
            out_evalues = compute_evalue(np.array(out_test_ce_loss_all_iters[iter][test_idx]), cal_set_ce_loss_all_iter[iter], thresholds[iter])
            out_evalues_all_traces.append(out_evalues)

            # Convert E-values to P-values
            out_p_values = evalue_to_pvalue(out_evalues, cal_evalues)
            out_p_values_all_traces.append(out_p_values)

        # Save Out-Distribution P-values
        np.savez(f"{opt.save_dir}/out_p_values_iter{iter+1}.npz", p_values=np.array(out_p_values_all_traces))

    print("All p-values and CE losses saved successfully.")

    

def calc_fisher_value(t_value, eval_n):
    summation = 0
    for i in range(eval_n): # calculating fisher value for the window in the datapoint
        summation += ((-np.log(t_value))**i)/np.math.factorial(i)
    return t_value*summation 

def calc_fisher_batch(p_values, eval_n): # p_values is 3D
    output = [[None]*len(window) for window in p_values[0]] # output is a 2D list for each datapoint, no of datapoints X number of windows in each datapoint
    for i in range(len(p_values[0])): #iterating over test datapoints
        for j in range(len(p_values[0][i])): #iterating over p-values for windows in the test datapoint
            prod = 1
            for k in range(eval_n):
                prod*=p_values[k][i][j][0]

            output[i][j] = calc_fisher_value(prod, eval_n)

    return output  # a 2D fisher value output for each window in each test datapoint

def eval_detection_fisher(eval_n):
    #pdb.set_trace()
    in_p = [] # 3D
    out_p = [] # 3D
    for iter in range(0, eval_n):
        in_p.append(np.load("{}/in_p_values_iter{}.npz".format(opt.save_dir, iter+1), allow_pickle=True)['p_values'])
        out_p.append(np.load("{}/out_p_values_iter{}.npz".format(opt.save_dir, iter+1), allow_pickle=True)['p_values'])

    in_fisher_values = calc_fisher_batch(in_p, eval_n) # a 2D fisher value output for each window in each iD test datapoint
    out_fisher_values = calc_fisher_batch(out_p, eval_n) # a 2D fisher value output for each window in each OOD test datapoint
    # pdb.set_trace()

    in_fisher_per_win = []
    for trace_idx in range(len(in_fisher_values)): # iterating over each iD trace
        for win_idx in range(len(in_fisher_values[trace_idx])): # iterating over each window in the trace
            in_fisher_per_win.append(in_fisher_values[trace_idx][win_idx])
    in_fisher_per_win = np.array(in_fisher_per_win)

    out_fisher_per_win = []
    for trace_idx in range(len(out_fisher_values)): # iterating over each OOD trace
        for win_idx in range(len(out_fisher_values[trace_idx])): # iterating over each window in the trace
            out_fisher_per_win.append(out_fisher_values[trace_idx][win_idx])
    out_fisher_per_win = np.array(out_fisher_per_win)

    np.savez("{}/in_fisher_iter{}.npz".format(opt.save_dir, iter+1), in_fisher_values_win=in_fisher_per_win)
    np.savez("{}/out_fisher_iter{}.npz".format(opt.save_dir, iter+1), out_fisher_values_win=out_fisher_per_win)

    #out_min_fisher_index_per_trace = [d.index(min(d)) for d in out_fisher_values]
    #print("Detection at frames: ", out_min_fisher_index_per_trace)
    # first_ood_frame_per_trace = [77, 46, 61, 50, 79, 64, 60, 57, 40, 57, 58, 46, 99, 86, 82, 83, 53, 54, 55, 46, 72, 57, 61, 42, 41, 56, 44, 36, 67, 70, 71, 50, 73, 85, 70, 53, 84, 79, 49, 78, 48, 81, 58, 43, 104, 72, 65, 65, 45, 87, 46, 39, 77, 50, 80, 38, 62, 59, 71, 61, 52, 49, 63, 52, 68, 82, 92, 66, 47, 53, 54, 55, 41] # the frame no. at which precipitation >= 20
    # print("Detection delay: ", np.array(out_min_fisher_index_per_trace)-np.array(first_ood_frame_per_trace))

    
    return in_fisher_per_win, out_fisher_per_win

def getAUROC(in_fisher_values, out_fisher_values):
    fisher_values = np.concatenate((in_fisher_values, out_fisher_values))

    indist_label = np.ones(len(in_fisher_values))
    ood_label = np.zeros(len(out_fisher_values))
    label = np.concatenate((indist_label, ood_label))

    from sklearn.metrics import roc_auc_score
    au_roc = roc_auc_score(label, fisher_values)*100
    return au_roc

def getTNR(in_fisher_values, out_fisher_values):

    in_fisher = np.sort(in_fisher_values)[::-1] # sorting in descending order
    tau = in_fisher[int(0.95*len(in_fisher))] # TNR at 95% TPR
    tnr = 100*(len(out_fisher_values[out_fisher_values<tau])/len(out_fisher_values))

    return tnr

if __name__ == "__main__":
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    auroc_all_trials = []
    # tnr_all_trials = []
    for trial in range(opt.trials):
        auroc_one_trial = []
        # tnr_one_trial = []
        checkOOD()
        for i in range(opt.n):
            print("Calculating fisher-values for n: ", i+1)
            in_fisher_values_per_win, out_fisher_values_per_win = eval_detection_fisher(i+1)
            au_roc = getAUROC(in_fisher_values_per_win, out_fisher_values_per_win)
            auroc_one_trial.append(au_roc)
            # tnr = getTNR(in_fisher_values_per_win, out_fisher_values_per_win)
            # tnr_one_trial.append(tnr)
            #print("For trial: {}, n: {}, AUROC: {}".format(trial+1, i+1, au_roc))
            # print("For trial: {}, n: {}, TNR: {}".format(trial+1, i+1, tnr))
        auroc_all_trials.append(auroc_one_trial)
        # tnr_all_trials.append(tnr_one_trial)

    auroc_all_trials = np.array(auroc_all_trials)
    # tnr_all_trials = np.array(tnr_all_trials)

    print("AUROC for CODiT(n=100) on {} as OOD data with window length {} is {}".format(opt.disease_type, opt.wl,np.mean(auroc_all_trials,0)[-1]))
    # print(np.std(auroc_all_trials,0))

    # print("TNR Mean: ", np.mean(tnr_all_trials,0))


    
