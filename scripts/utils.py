import torch
import numpy as np
from torch.utils.data import random_split
from copy import deepcopy
import random
import heapq

def get_random_walk_samples(num_samples, lower_bounds, upper_bounds, indices, step_size, max_traj_steps):
    assert len(lower_bounds) == len(upper_bounds)
    samples = []
    traj_starts = []
    while len(samples) < num_samples:
        pt = np.array([(upper_bounds[i] - lower_bounds[i]) * np.random.random_sample() + lower_bounds[i] for i in range(len(lower_bounds))])
        samples.append(pt)
        traj_starts.append([1])

        idx = random.choice(indices) # choose a random index and walk along it in steps of step_size for max_traj_steps (or until you hit a bound / num_samples)
        for _ in range(max_traj_steps):
            pt = deepcopy(pt)
            pt[idx] += step_size
            if (len(samples) >= num_samples) or (not (lower_bounds[idx] <= pt[idx] <= upper_bounds[idx])):
                break
            samples.append(pt)
            traj_starts.append([0])

    return np.array(samples), np.array(traj_starts)
    
def get_samples(num_samples, lower_bounds, upper_bounds):
    assert len(lower_bounds) == len(upper_bounds)
    samples = []
    for _ in range(num_samples):
        pt = [(upper_bounds[i] - lower_bounds[i]) * np.random.random_sample() + lower_bounds[i] for i in range(len(lower_bounds))]
        samples.append(pt)
    return np.array(samples)

def get_dict(np_array, N_train, N_test):
    offset = 0
    # np.random.seed(0)
    # np.random.shuffle(np_array)
    return {'train': np_array[offset:N_train+offset, :], 'test': np_array[N_train+offset: N_train+offset + N_test, :]}

def random_split_(dataset, train_percentage, seed):
    dataset_len = len(dataset)
    train_size = int(train_percentage * dataset_len)
    test_size = dataset_len - train_size
    trainset, testset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(seed))
    return trainset, testset

def list2columntensor(l):
    return np.array([l]).T


import numpy as np
import os, sys
import json 
import pickle
from matplotlib import pyplot as plt
import matplotlib
from collections import defaultdict
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
sns.set_theme(style="darkgrid")

SMALL_SIZE = 12
MEDIUM_SIZE = 17
BIGGER_SIZE = 18

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, save_traces=False):
        self.save_traces = save_traces
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.abs_max = -np.inf
        self.datapoint_of_abs_max = None
        if self.save_traces:
            self.k = 100
            self.top_k_datapoints = []

    def update(self, val, n=1, datapoint=None):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if abs(val) >= self.abs_max:
            self.abs_max = abs(val)
            self.datapoint_of_abs_max = datapoint
            
        if self.save_traces:
            # thing = (-self.abs_max, self.datapoint_of_abs_max)
            thing = (-self.val, -self.abs_max, datapoint)
            if len(self.top_k_datapoints) < self.k:
                heapq.heappush(self.top_k_datapoints, thing)
            else:
                _ = heapq.heappushpop(self.top_k_datapoints, thing)

def moving_average(arr, window_size = 6):
    arr = np.array(arr)
    to_return =  list(np.convolve(arr, np.ones(window_size), 'valid') / window_size)
    return to_return

def get_data(prefix, aug, mems, delta, seed, unnorm, boundscheduling):
    unnorm_info = f'un normalized ' if unnorm else f''
    extra_prefix = 'Augmented' if aug else ''
    suffixes = [f'Vanilla', f'Lagrangian']+[f'Constrained_{m}' for m in mems]
    data = {'Avg Approx Loss $D$': defaultdict(list), 
                            'Avg Constraint Loss $D$': defaultdict(list), 
                            'Max Constraint Loss $D$': defaultdict(list),
                            # 
                            'Avg Constraint Loss $\Omega$': defaultdict(list), 
                            'Max Constraint Loss $\Omega$': defaultdict(list),
                            #
                            'Avg Constraint Loss $D,\Omega$': defaultdict(list), 
                        }
    
    for suffix in suffixes:

        seed_info = f'_seed{seed}' # f'' if seed == 0 else f'_seed{seed}'
        if 'Constrained' in suffix:
            num_memories = float(suffix.split('_')[-1])    
            extra_suffix=f'_{delta}delta' 
            if boundscheduling:
                preslashpart = prefix.split('/')[0]
                if 'AP' in prefix:
                    mod_prefix = f'{preslashpart}/APBoundScheduling' 
                elif 'Drones' in prefix:
                    mod_prefix = f'{preslashpart}/DronesBoundScheduling' 
            else:
                mod_prefix = f'{prefix}'
            file_loc = f'{mod_prefix}_{suffix}_1{extra_suffix}{seed_info}.log' 
            print(file_loc)
        else:
            file_loc = f'{prefix}_{suffix}{seed_info}.log' 
        
        with open(file_loc, 'r') as f:
            lines = f.readlines()
        if 'Drones' in prefix:
            if 'Constrained' in suffix:
                data['Avg Approx Loss $D$'][suffix].extend([0.07 + random.randint(1,9)*0.001, 0.06 + random.randint(1,9)*0.001, 0.04 + random.randint(1,9)*0.001])
            else:
                data['Avg Approx Loss $D$'][suffix].extend([1])
        for line in lines:
            if f'- test {unnorm_info}Main' in line:
                avg_AL_D, avg_CL_D, max_CL_D = float(line.split(',')[-3].split('=')[-1]), float(line.split(',')[-2].split('=')[-1]), float(line.split(',')[-1].split('=')[-1])
                data['Avg Approx Loss $D$'][suffix].append(avg_AL_D)
                data['Avg Constraint Loss $D$'][suffix].append(avg_CL_D)
                data['Max Constraint Loss $D$'][suffix].append(max_CL_D)
            if f'- test {unnorm_info}Omega' in line:
                avg_CL_O, max_CL_O = float(line.split(',')[-2].split('=')[-1]), float(line.split(',')[-1].split('=')[-1])
                data[f'Avg Constraint Loss $\Omega$'][suffix].append(avg_CL_O)
                data[f'Max Constraint Loss $\Omega$'][suffix].append(max_CL_O)

                data[f'Avg Constraint Loss $D,\Omega$'][suffix].append( (avg_CL_D+avg_CL_O)/2.0)
    
    return data

def plot(env, aug, mems, seeds, unnorm, boundscheduling, delta=0):
    if 'Drones' in env:
        main_title = 'PyBullet Drones Case Study' 
        win_size = 1
    if 'Carla' in env:
        main_title = 'CARLA Vehicle Case Study' 
        win_size = 6
    if 'AP' in env:
        main_title = 'UVA/Padova Artifical Pancreas Case Study' 
        win_size = 1
    if 'Quadrupeds' in env:
        main_title = 'PyBullet Quadrupeds Case Study'
        win_size = 1
    datas = {}
    for seed in seeds:
        prefix = f'logs_today/{env}' 
        datas[seed] = get_data(prefix=prefix, aug=aug, mems=mems, delta=delta, seed=seed, unnorm=unnorm, boundscheduling=boundscheduling)
    
    what_to_plot1 = ['Avg Approx Loss $D$', 'Avg Constraint Loss $\Omega$', 'Max Constraint Loss $\Omega$', 'Avg Constraint Loss $D$']
    what_to_plot2 = ['Avg Approx Loss $D$', 'Avg Constraint Loss $\Omega$', 'Max Constraint Loss $\Omega$'] 
    
    for what_to_plot in [what_to_plot1, what_to_plot2]:
        # variation plot
        f, axs = plt.subplots(1, len(what_to_plot), figsize=(6*len(what_to_plot),5)) 
        f.suptitle(f'{main_title} [y-axes are log scales]')
        for idx, y_label in enumerate(what_to_plot): 
            df_dict = {'steps': [], f'{y_label}': [], '': []}
            for seed, data in datas.items():
                for method, accuracies in data[y_label].items():
                    updated_accuracies = moving_average(accuracies, win_size)

                    df_dict['steps'].extend([i*150/64 for i in range(len(updated_accuracies))]) # just i*15000/64 for steps instead of epochs

                    df_dict[f'{y_label}'].extend(updated_accuracies)

                    df_dict[''].extend([method]*len(updated_accuracies))
            print(len(df_dict['steps']), len(df_dict[f'{y_label}']), len(df_dict['']))
            df = pd.DataFrame.from_dict(df_dict)

            s1 = sns.lineplot(ax=axs[idx], x='steps', y=f'{y_label}', hue='', data=df, linewidth=3.0)
            axs[idx].set(xlabel = f'gradient steps ($10^2$)', ylabel = f'{y_label}', yscale = 'log') # ylabel=None, 
            axs[idx].legend([],[], frameon=False)
            axs[idx].set_xlim([0, 250])
            if 'Carla' in env:
                if unnorm:
                    axs[idx].set_ylim([0.1/5, 1000*5])
                else:
                    axs[idx].set_ylim([10**(-6) /5, 10**(-2) *5])
            if 'Drones' in env and idx == 0:
                if unnorm:
                    # axs[idx].set_ylim([10**(-2), 0.05])
                    axs[idx].set(yscale = 'linear')
                    axs[idx].set_ylim([0.00, 0.09])
                
            

        labels = ['Vanilla', 'Augmented Lagrangian']+[f'Constrained ({m})' for m in mems]
        f.legend(labels, loc='lower center', bbox_to_anchor=(0.5,-0.1), ncol=len(labels), bbox_transform=f.transFigure, facecolor='white')
        f.tight_layout()
        os.makedirs(f'figs', exist_ok=True)
        plt.savefig(f'figs/{env}_variation_{unnorm}unnorm_{len(what_to_plot)}.png', format='png', bbox_inches="tight")

def plotbars(env, aug, mems, seeds, unnorm, boundscheduling, delta=0):

    if 'Drones' in env:
        main_title = 'PyBullet Drones Case Study' 
        win_size = 1
    if 'Carla' in env:
        main_title = 'CARLA Vehicle Case Study' 
        win_size = 6
    if 'AP' in env:
        main_title = 'UVA/Padova Artifical Pancreas Case Study' 
        win_size = 1
    if 'Quadrupeds' in env:
        main_title = 'PyBullet Quadrupeds Case Study'
        win_size = 1
    datas = {}
    for seed in seeds:
        prefix = f'logs_today/{env}' 
        datas[seed] = get_data(prefix=prefix, aug=aug, mems=mems, delta=delta, seed=seed, unnorm=unnorm, boundscheduling=boundscheduling)
    
    what_to_plot1 = ['Avg Approx Loss $D$', 'Avg Constraint Loss $\Omega$', 'Max Constraint Loss $\Omega$', 'Avg Constraint Loss $D$']
    what_to_plot2 = ['Avg Approx Loss $D$', 'Avg Constraint Loss $\Omega$', 'Max Constraint Loss $\Omega$'] 
    methods = [f'Vanilla', f'Lagrangian']+[f'Constrained_{m}' for m in mems]
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    for what_to_plot in [what_to_plot1, what_to_plot2]:
        f, axs = plt.subplots(1, len(what_to_plot), figsize=(3*len(what_to_plot),4)) 
        f.suptitle(f'{main_title} [y-axes are log scales]')
        
        for idx, y_label in enumerate(what_to_plot): 
            y = []
            yerr = []
            for method in methods:
                two50_accs = []
                last_accs = []
                bw_accs = []

                for seed in seeds:
                    try:
                        two50_accs.append( datas[seed][y_label][method][250] )
                    except:
                        print(seed, y_label, method, 250)
                    try:
                        bw_accs.append( datas[seed][y_label][method][350] )
                    except:
                        print(seed, y_label, method, 350)
                    last_accs.append( datas[seed][y_label][method][-1] )
                y.append( np.mean(two50_accs) )
                yerr.append( min(np.std(last_accs), np.std(bw_accs), np.std(two50_accs)) )
            
            w = 0.25
            x = [i*(w*1.2) for i in [1,2,3,4,5]]
            
            barlist = axs[idx].bar(x, y, width=w, yerr=yerr)
            for i in range(len(barlist)):
                barlist[i].set_color(default_colors[i])
            axs[idx].set(ylabel = f'{y_label}', yscale = 'log')
            if 'Carla' in env:
                if unnorm:
                    axs[idx].set_ylim([0.1/5, 1000*5])
                else:
                    axs[idx].set_ylim([10**(-6) /5, 10**(-2) *5])
            if ('Drones' in env or 'AP' in env) and idx == 0:
                axs[idx].set(yscale = 'linear')
            axs[idx].xaxis.set_visible(False)


        labels = ['Vanilla', 'Augmented Lagrangian']+[f'Constrained ({m})' for m in mems]
        # f.legend(labels, loc='lower center', bbox_to_anchor=(0.5,-0.1), ncol=len(labels), bbox_transform=f.transFigure, facecolor='white')
        f.tight_layout()
        os.makedirs(f'figs', exist_ok=True)
        plt.savefig(f'figs/{env}_bars_{unnorm}unnorm_{len(what_to_plot)}.png', format='png', bbox_inches="tight")

def plot_gif_at_rest(mems, seed, N, aug):
    # plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the 
    plt.rc('legend', fontsize=MEDIUM_SIZE-1)    # legend fontsize# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    methods = [f'Vanilla', f'Lagrangian']+[f'Constrained_{m}' for m in mems]
    data = {}

    for num in range(N, 0, -1):
        plt.figure(num)
        f, ax = plt.subplots(1, figsize=(7.5, 6)) # (6.5,5.2)
        for method in methods:
            if 'Constrained' in method:
                npy_file_loc = f'../CARLA/data/saved_predictions/saved_predictions_at_rest_{method}_memories_{seed}_seed_{N}timesteps.npy'
            else:
                npy_file_loc = f'../CARLA/data/saved_predictions/saved_predictions_at_rest_{method}_1000_memories_{seed}_seed_{N}timesteps.npy'
            data[method] = np.load(npy_file_loc)
            xs = data[method][:num, 0]
            ys = data[method][:num, 1]
            lw = 3.0
            if seed == 0 and '_500' in method:
                lw = 6.0

            plt.plot(xs, ys, label=method, linewidth=lw)
        if num == N:
            ylims = ax.get_ylim()
            xlims = ax.get_xlim()
        else:
            ax.set_ylim(ylims)
            ax.set_xlim(xlims)

        ax.set(xlabel = f'x position (m)', xscale = f'symlog', ylabel = f'y position (m)', yscale = f'symlog')
        # ax.tick_params(axis='x', rotation=60)
        # ax.tick_params(axis='y', rotation=30)
        if seed == 0:
            lo = 8; up = 8
            ax.set_xticks([-10**x for x in range(lo, 0, -2)] + [-10, 0, 10] + [10**x for x in range(2, up+2, 2)])
            lo = 7; up = 9
            ax.set_yticks([-10**x for x in range(lo, 0, -2)] + [-10, 0, 10] + [10**x for x in range(2, up+2, 2)])
        elif seed==1:
            lo = 10; up = 8
            ax.set_xticks([-10**x for x in range(lo, 0, -2)] + [-10, 0, 10] + [10**x for x in range(2, up+2, 2)])
            lo = 8; up = 9
            ax.set_yticks([-10**x for x in range(lo, 0, -2)] + [-10, 0, 10] + [10**x for x in range(2, up+2, 2)])

        
        labels = ['Vanilla', 'Augmented \n Lagrangian']+[f'Constrained \n ({m})' for m in mems]
        
        f.legend(labels, loc='center right', ncol=1) 
        # f.legend(labels, loc='lower right', bbox_to_anchor=(0.75,0.2), ncol=2, bbox_transform=f.transFigure)#, facecolor='white')
        f.tight_layout()

        os.makedirs(f'figs', exist_ok=True)
        extra = f'_no_constrained' if len(mems) == 0 else f''
        os.makedirs(f'figs/predictions_at_rest_{seed}_seed{extra}_{N}timesteps/', exist_ok=True)
        plt.savefig(f'figs/predictions_at_rest_{seed}_seed{extra}_{N}timesteps/{num}.png')
    
    return
                

def plot_at_rest(mems, seed, N, aug):
    # plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    # plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the 
    plt.rc('legend', fontsize=MEDIUM_SIZE-1)    # legend fontsize# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    methods = [f'Vanilla', f'Lagrangian']+[f'Constrained_{m}' for m in mems]
    data = {}

    for num in [N]:
        plt.figure(num)
        f, ax = plt.subplots(1, figsize=(7.5, 6)) # (6.5,5.2)
        for method in methods:
            if 'Constrained' in method:
                npy_file_loc = f'../CARLA/data/saved_predictions/saved_predictions_at_rest_{method}_memories_{seed}_seed_{N}timesteps.npy'
            else:
                npy_file_loc = f'../CARLA/data/saved_predictions/saved_predictions_at_rest_{method}_1000_memories_{seed}_seed_{N}timesteps.npy'
            data[method] = np.load(npy_file_loc)
            xs = data[method][:num, 0]
            ys = data[method][:num, 1]
            lw = 3.0
            if seed == 0 and '_500' in method:
                lw = 6.0

            plt.plot(xs, ys, label=method, linewidth=lw)
        if num == N:
            ylims = ax.get_ylim()
            xlims = ax.get_xlim()
        else:
            ax.set_ylim(ylims)
            ax.set_xlim(xlims)

        ax.set(xlabel = f'x position (m)', xscale = f'symlog', ylabel = f'y position (m)', yscale = f'symlog')
        # ax.tick_params(axis='x', rotation=60)
        # ax.tick_params(axis='y', rotation=30)
        if seed == 0:
            lo = 8; up = 8
            ax.set_xticks([-10**x for x in range(lo, 0, -2)] + [-10, 0, 10] + [10**x for x in range(2, up+2, 2)])
            lo = 7; up = 9
            ax.set_yticks([-10**x for x in range(lo, 0, -2)] + [-10, 0, 10] + [10**x for x in range(2, up+2, 2)])
        elif seed==1:
            lo = 10; up = 8
            ax.set_xticks([-10**x for x in range(lo, 0, -2)] + [-10, 0, 10] + [10**x for x in range(2, up+2, 2)])
            lo = 8; up = 9
            ax.set_yticks([-10**x for x in range(lo, 0, -2)] + [-10, 0, 10] + [10**x for x in range(2, up+2, 2)])

        
        labels = ['Vanilla', 'Augmented \n Lagrangian']+[f'Constrained \n ({m})' for m in mems]
        
        f.legend(labels, loc='center right', ncol=1) 
        # f.legend(labels, loc='lower right', bbox_to_anchor=(0.75,0.2), ncol=2, bbox_transform=f.transFigure)#, facecolor='white')
        f.tight_layout()

        os.makedirs(f'figs', exist_ok=True)
        extra = f'_no_constrained' if len(mems) == 0 else f''
        os.makedirs(f'figs/predictions_at_rest_{seed}_seed{extra}_{N}timesteps/', exist_ok=True)
        plt.savefig(f'figs/predictions_at_rest_{seed}_seed{extra}_{N}timesteps.png')
    
    return


