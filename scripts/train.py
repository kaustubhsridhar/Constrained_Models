import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from model import *
from dataset import *
from utils import AverageMeter, random_split_
from tqdm import tqdm
import numpy as np
import argparse 
import random

parser = argparse.ArgumentParser(description='Parse constants.')
parser.add_argument('--no_tqdm', default=False, action='store_true', help='set if are logging to file instead of terminal')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--gpu', default=0, help='0, 1')
parser.add_argument('--env', default='Drones', help='Carla, Drones, AP, Quadrupeds')
parser.add_argument('--method', default='Constrained', help='Vanilla, Lagrangian, Constrained')
parser.add_argument('--num_memories', default=1000, type=int, help='maximum number of memories')
parser.add_argument('--num_robots', default=6, type=int, help='to read data for how many robots in env?')
parser.add_argument('--delta', default=0.0, type=float, help='delta around voronoi bounds')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate for Adam optimize')
parser.add_argument('--gng_epochs', default=1, type=int, help='num epochs for gng')
parser.add_argument('--num_voronoi_samples', default=25, type=int, help='num of points to sample in each voronoi cell')
parser.add_argument('--epochs', default=500, type=int, help='num epochs')
parser.add_argument('--epsilon', default=0.01, type=float, help='perturbation')
parser.add_argument('--batch_size', default=64, type=int, help='train and test (not adv_test) batch_size')
parser.add_argument('--self_sup', default=False, action='store_true', help='set for self supervised training')
parser.add_argument('--only_eval', default=False, action='store_true', help='set if you only want to eval trained models')
parser.add_argument('--augmented', default=False, action='store_true', help='set if you want augmented lagrangian instead of fixed lagrangian')
args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

if args.method == 'Constrained':
    trainset, testset, trainset_Omega, testset_Omega = make_BoundsAndDynamicsDataset_instance(args.env, max_memories=args.num_memories, gng_epochs=args.gng_epochs, NumRobotsInEnv=args.num_robots, num_voronoi_samples=args.num_voronoi_samples, delta=args.delta, seed=args.seed)
else:
    trainset, testset, trainset_Omega, testset_Omega = make_LagrangianDataset_instance(args.env, NumRobotsInEnv=args.num_robots, seed=args.seed)
input_size = trainset.input_size
output_size = trainset.output_size

trainloader = DataLoader(trainset, args.batch_size, shuffle=False)
testloader = DataLoader(testset, 1, shuffle=False)
trainloader_Omega = DataLoader(trainset_Omega, args.batch_size, shuffle=False)
testloader_Omega = DataLoader(testset_Omega, 1, shuffle=False)

if args.env == 'AP':
    model = prediction_model_AP(input_size, output_size)
elif args.env == 'Quadrupeds':
    model = prediction_model(input_size, output_size) # _quadruped
else:
    model = prediction_model(input_size, output_size)
model.to(device)

os.makedirs(f'{trainset.dir}/saved_models', exist_ok=True)
extra_info = f'_selfsup' if args.self_sup else f''
model_filename = f'{trainset.dir}/saved_models/best{extra_info}_{args.method}_net_{args.num_memories}_memories_{args.seed}_seed'
if not args.only_eval:
    if args.self_sup:
        print(f'Self-supervised training...')
        vanilla_model = prediction_model(input_size, output_size)
        if not args.method == 'Vanilla':
            vanilla_model.load_state_dict(torch.load(f'{trainset.dir}/saved_models/best_Vanilla_net_1000_memories', map_location = device))
        vanilla_model.to(device)
        
    train(model=model, 
        trainloader=trainloader, 
        testloader=testloader, 
        trainloader_Omega=trainloader_Omega, 
        testloader_Omega=testloader_Omega, 
        epochs=args.epochs, 
        learning_rate=args.lr, 
        device=device, 
        filename=model_filename, 
        method=args.method, 
        env=args.env,
        no_tqdm=args.no_tqdm,
        eps=args.epsilon, 
        family_of_dynamics=trainset.family_of_dynamics, 
        num_control_inputs=trainset.num_control_inputs, 
        voronoi_bounds=trainset.voronoi_bounds if args.method == 'Constrained' else None,
        gng=trainset.gng if args.method == 'Constrained' else None,
        test_every_N_epochs=1,
        augmented=args.augmented,
        vanilla_model=vanilla_model if args.self_sup else None,
        allow_omega_approx_loss=False,
        output_un_normalizer=trainset.output_un_normalizer,
        )
else:
    if args.env == 'AP':
        model.load_state_dict(torch.load(model_filename, map_location=device))
        test(model=model, testloader=testloader_Omega, device=device, method=args.method, no_tqdm=args.no_tqdm, testloader_type='Omega', un_normalizer=trainset.output_un_normalizer, print_traces=True, input_un_normalizer=trainset.input_un_normalizer,
            voronoi_bounds=trainset.voronoi_bounds if args.method == 'Constrained' else None,
            gng=trainset.gng if args.method == 'Constrained' else None,)
    elif args.env == 'Carla':
        model.load_state_dict(torch.load(model_filename, map_location=device))
        os.makedirs(f'{trainset.dir}/saved_predictions/', exist_ok=True)
        for N in [20, 50]:
            test_at_rest_for_N_timesteps(model=model, testloader=testloader_Omega, device=device, method=args.method, no_tqdm=args.no_tqdm, testloader_type='Omega', un_normalizer=trainset.output_un_normalizer, print_traces=True,
                voronoi_bounds=trainset.voronoi_bounds if args.method == 'Constrained' else None,
                gng=trainset.gng if args.method == 'Constrained' else None, N=N, 
                saveloc = f'{trainset.dir}/saved_predictions/saved_predictions_at_rest_{args.method}_{args.num_memories}_memories_{args.seed}_seed_{N}timesteps.npy')


