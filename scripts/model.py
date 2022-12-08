import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import AverageMeter
from tqdm import tqdm
import numpy as np
import random
import pickle 

class prediction_model(nn.Module):
    def __init__(self, input_size, output_size):
        super(prediction_model, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        w1 = 1024
        w2 = 1024
        w3 = 128

        self.linear1 = nn.Linear(self.input_size, w1)
        self.linear6 = nn.Linear(w1, self.output_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = self.linear6(output)
        return output

class prediction_model_quadruped(nn.Module):
    def __init__(self, input_size, output_size):
        super(prediction_model_quadruped, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        w1 = 1024
        w2 = 1024

        self.linear1 = nn.Linear(self.input_size, w1)
        self.linear2 = nn.Linear(w1, w1)
        self.linear3 = nn.Linear(w1, self.output_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        return output

class prediction_model_AP(nn.Module):
    def __init__(self, input_size, output_size):
        super(prediction_model_AP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        w1 = 20
        w2 = 10

        self.linear1 = nn.Linear(self.input_size, w1)
        self.linear2 = nn.Linear(w1, w2)
        self.linear3 = nn.Linear(w2, self.output_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        return output

def alt_sigmoid(x):
    return nn.Sigmoid()(x)


def train(model, trainloader, testloader, trainloader_Omega, testloader_Omega, epochs, learning_rate, device, filename, method, env, no_tqdm, eps, family_of_dynamics, num_control_inputs, voronoi_bounds, gng, test_every_N_epochs=1, augmented=False, vanilla_model=None, allow_omega_approx_loss=False, output_un_normalizer=1): 
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.MSELoss()
    test_losses = []
    best = np.inf
    self_sup = False if (vanilla_model is None) else True
    lamda = 0.01 if (env == 'Quadrupeds') else 1
    if augmented:
        lamda = 0
        mu = 1e-3
        mu_mult = 1.5
    BoundScheduling = True if ((env == 'AP') and method == 'Constrained') else False #  or env == 'Drones'
    k = 0.5

    for epoch in range(epochs):
        testloader_Omega_type = 'Main2' if allow_omega_approx_loss else 'Omega'
        for tl, tl_type in [(testloader, 'Main'), (testloader_Omega, testloader_Omega_type)]:
            avg_test_AL, avg_test_CL, max_test_CL, avg_test_AL_un_norm, avg_test_CL_un_norm, max_test_CL_un_norm = test(model, tl, device, method, no_tqdm, tl_type, output_un_normalizer)
            test_losses.append([avg_test_AL, avg_test_CL, max_test_CL]) 
            print(f'------------------------- test {tl_type}, {avg_test_AL=}, {avg_test_CL=}, {max_test_CL=}')
            print(f'------------------------- test un normalized {tl_type}, {avg_test_AL_un_norm=}, {avg_test_CL_un_norm=}, {max_test_CL_un_norm=}')

        approx_losses = AverageMeter()
        constraint_losses = AverageMeter()

        loader = zip(trainloader, trainloader_Omega)
        loader_type = 'Both'
        
        progress_bar = tqdm(loader, dynamic_ncols=True) if not no_tqdm else loader
        for idx, (D_batch, Omega_batch) in enumerate(progress_bar):
            i, t, _, lo, up, fam = D_batch
            i_O, t_O, _, lo_O, up_O, fam_O = Omega_batch

            if BoundScheduling:
                lo = lo - (up-lo) * k; lo_O = lo_O - (up_O-lo_O) * k
                up = up + (up-lo) * k; up_O = up_O + (up_O-lo_O) * k
            
            # move to device and make floats
            i, t, lo, up, fam, i_O, t_O, lo_O, up_O, fam_O = i.to(device).float(), t.to(device).float(), lo.to(device).float(), up.to(device).float(), fam.to(device).float(), i_O.to(device).float(), t_O.to(device).float(), lo_O.to(device).float(), up_O.to(device).float(), fam_O.to(device).float()
            
            # pseudo targets if sel sup for Omega data only
            if self_sup:
                with torch.no_grad():
                    t_O = vanilla_model(i_O) # Pseudo next_states

            # forward
            if method == 'Constrained':
                out = lo + alt_sigmoid(model(i)) * (up - lo)
                out_O = lo_O + alt_sigmoid(model(i_O)) * (up_O - lo_O)
            else:
                out = model(i)
                out_O = model(i_O)

            # approx loss (AL)
            approx_loss = criterion(out, t)
            if self_sup or allow_omega_approx_loss:
                approx_loss += criterion(out_O, t_O)
            approx_losses.update(approx_loss.item(), i.shape[0])

            # constraint loss (CL)
            CL_list = [criterion(out, fam[:, didx, :]) for didx in range(fam.shape[1])]
            CL_list.extend([criterion(out_O, fam_O[:, didx, :]) for didx in range(fam_O.shape[1])])
            CL = torch.stack(CL_list, axis=0).mean()
            
            constraint_losses.update(CL.item(), i.shape[0])

            # final loss
            if method == 'Vanilla':
                loss = approx_loss
            else:
                loss = approx_loss + lamda * CL

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not no_tqdm: progress_bar.set_postfix({f'train {loader_type} ep': epoch, 'approx_loss': approx_losses.avg, 'constraint_loss': constraint_losses.avg}) 
        print(f'train {epoch=}, {loader_type=}, {approx_losses.avg=}, {constraint_losses.avg=}')

        if epoch%test_every_N_epochs == 0:
            if augmented:
                lamda = abs(lamda + 2 * mu * constraint_losses.avg)
                mu = mu * mu_mult

            if BoundScheduling:
                k = k * (0.99**(epoch+1))

            testloader_Omega_type = 'Main2' if allow_omega_approx_loss else 'Omega'
            for tl, tl_type in [(testloader, 'Main'), (testloader_Omega, testloader_Omega_type)]:
                avg_test_AL, avg_test_CL, max_test_CL, avg_test_AL_un_norm, avg_test_CL_un_norm, max_test_CL_un_norm = test(model, tl, device, method, no_tqdm, tl_type, output_un_normalizer)
                test_losses.append([avg_test_AL, avg_test_CL, max_test_CL]) 
                print(f'------------------------- test {tl_type}, {avg_test_AL=}, {avg_test_CL=}, {max_test_CL=}')
                print(f'------------------------- test un normalized {tl_type}, {avg_test_AL_un_norm=}, {avg_test_CL_un_norm=}, {max_test_CL_un_norm=}')

            if max_test_CL <= best:
                best = avg_test_AL
                torch.save(model.state_dict(), filename)

    return np.array(test_losses)

def test(model, testloader, device, method, no_tqdm, testloader_type, un_normalizer, print_traces=False, input_un_normalizer=None, voronoi_bounds=None, gng=None):
    criterion = nn.MSELoss()
    with torch.no_grad():
        approx_losses = AverageMeter()
        constraint_losses = AverageMeter()
        un_normalizer = torch.from_numpy(un_normalizer).to(device)
        input_un_normalizer = torch.from_numpy(input_un_normalizer).to(device) if input_un_normalizer is not None else 1 # for printing traces only
        approx_losses_un_normalized = AverageMeter()
        constraint_losses_un_normalized = AverageMeter(save_traces=print_traces)

        if not no_tqdm:
            progress_bar = tqdm(testloader, dynamic_ncols=True)
            progress_bar.set_description(f'test')
        else:
            progress_bar = testloader

        for idx, (inputs, targets, is_traj_starting, lower_bounds, upper_bounds, family_of_dynamics_targets) in enumerate(progress_bar):
            inputs, targets, lower_bounds, upper_bounds, family_of_dynamics_targets = inputs.to(device).float(), targets.to(device).float(), lower_bounds.to(device).float(), upper_bounds.to(device).float(), family_of_dynamics_targets.to(device).float()

            if method == 'Constrained':
                outputs = lower_bounds + alt_sigmoid(model(inputs)) * (upper_bounds - lower_bounds)
            else:
                outputs = model(inputs)
            
            num_dynamics_in_family = family_of_dynamics_targets.shape[1] # family_of_dynamics_targets.shape == (batch_size, num_dynamics_in_family, num_states)
            constraint_loss_list = [criterion(outputs, family_of_dynamics_targets[:, didx, :]) for didx in range(num_dynamics_in_family)]
            constraint_loss = torch.stack(constraint_loss_list, axis=0).mean()

            constraint_loss_list_un_normalized = [criterion(outputs * un_normalizer, family_of_dynamics_targets[:, didx, :] * un_normalizer) for didx in range(num_dynamics_in_family)]
            constraint_loss_un_normalized = torch.stack(constraint_loss_list_un_normalized, axis=0).mean()
            
            if not testloader_type == 'Omega':
                approx_loss = criterion(outputs, targets)
                approx_losses.update(approx_loss.item(), inputs.shape[0])
                
                approx_loss_un_normalized = criterion(outputs * un_normalizer, targets * un_normalizer)
                approx_losses_un_normalized.update(approx_loss_un_normalized.item(), inputs.shape[0])

            constraint_losses.update(constraint_loss.item(), inputs.shape[0], datapoint=(inputs, outputs, family_of_dynamics_targets, lower_bounds, upper_bounds))
            
            constraint_losses_un_normalized.update(constraint_loss_un_normalized.item(), inputs.shape[0], datapoint=(inputs, outputs, family_of_dynamics_targets, lower_bounds, upper_bounds))
            if not no_tqdm:
                progress_bar.set_postfix({'approx_loss': approx_losses.avg, 'constraint_loss': constraint_losses.avg}) 
    
    if print_traces:
        traces = constraint_losses_un_normalized.top_k_datapoints
        # traces = constraint_losses_un_normalized.all
        delta_monotonicity_testing(traces, model, method, un_normalizer, input_un_normalizer, device, voronoi_bounds, gng)
    
    # print(f'Point where abs_max is observed is \n {constraint_losses.datapoint_of_abs_max} \n\n')
    return approx_losses.avg, constraint_losses.avg, constraint_losses.abs_max, approx_losses_un_normalized.avg, constraint_losses_un_normalized.avg, constraint_losses_un_normalized.abs_max

def delta_monotonicity_testing(traces, model, method, un_normalizer, input_un_normalizer, device, voronoi_bounds, gng):
    max_increase = -np.inf
    num_increases = 0
    num_total = 0
    sum_increase = 0
    
    # if method == 'Lagrangian':
    #     with open(f'./temp_top_10_traces_of_Lagrangian.pkl', 'wb') as f:
    #         pickle.dump(traces, f)
    # elif method == 'Constrained':
    #     with open(f'./temp_top_10_traces_of_Lagrangian.pkl', 'rb') as f:
    #         traces = pickle.load(f)
    with torch.no_grad():
        for item in traces:
            tup = item[-1]
            inputs = tup[0]
            outputs = tup[1]
            dynamics_target = tup[2]
            lower_bounds = tup[3]
            upper_bounds = tup[4]

            idx = 10 + random.randint(0, 9)
            adv_inputs = inputs.clone()
            adv_inputs[:, idx] = inputs[:, idx] + random.randint(6,9)/30.0

            if method == 'Constrained':
                adv_lower_bounds, adv_upper_bounds = get_voronoi_then_bounds(adv_inputs.detach().cpu().numpy(), voronoi_bounds, gng)
                adv_lower_bounds, adv_upper_bounds = torch.from_numpy(adv_lower_bounds).to(device), torch.from_numpy(adv_upper_bounds).to(device)
                adv_outputs = adv_lower_bounds + alt_sigmoid(model(adv_inputs)) * (adv_upper_bounds - adv_lower_bounds)
            else:
                adv_outputs = model(adv_inputs)

            # print(f'{method=}')
            # print(f'{inputs*input_un_normalizer=}')
            # print(f'with insulin increased at {idx=} to -->')
            # print(f'{adv_inputs*input_un_normalizer=}')
            # print(f'with outputs which should have decreased from {outputs*un_normalizer=} --> {adv_outputs*un_normalizer} \n')
        
            diff = (adv_outputs.item() - outputs.item())*un_normalizer
            num_total += 1
            if diff > 0:
                max_increase = max(max_increase, diff)
                sum_increase += diff
                num_increases += 1

    print(f'{method=}, {max_increase=}, {num_increases=}, {num_total=}, {num_increases/num_total=}, {sum_increase/num_total=}')
    return 

from neural_gas_helpers import L2dist, m_step_neighbours
def get_voronoi_then_bounds(input, voronoi_bounds, gng):
    nodes_to_search = voronoi_bounds['lo'].keys()

    nodes_sorted_by_dist = sorted([(node, L2dist(input, node.weight.flatten())) for node in nodes_to_search], key= lambda tup:tup[1])    
    closest_node = nodes_sorted_by_dist[0][0]
    
    lower_bound = voronoi_bounds['lo'][closest_node]
    upper_bound = voronoi_bounds['up'][closest_node]

    return lower_bound, upper_bound


def test_at_rest_for_N_timesteps(model, testloader, device, method, no_tqdm, testloader_type, un_normalizer, saveloc, print_traces=False, input_un_normalizer=None, voronoi_bounds=None, gng=None, N=10):
    with torch.no_grad():
        un_normalizer = torch.from_numpy(un_normalizer).to(device).reshape((1,3))
        # input_un_normalizer = torch.from_numpy(input_un_normalizer).to(device) if input_un_normalizer is not None else 1 # for printing traces only

        initial_state = np.array([[0, 0, 0]])
        outputs = torch.from_numpy(initial_state).to(device).float()
        zero_controls = torch.from_numpy(np.array([[0, 0]])).to(device).float()
        all_states = [initial_state[0]]
        for timestep in range(N): 
            inputs = torch.cat( (outputs.clone(), zero_controls.clone()), axis=1)

            if method == 'Constrained':
                lower_bounds, upper_bounds = get_voronoi_then_bounds(inputs.detach().cpu().numpy(), voronoi_bounds, gng)
                lower_bounds, upper_bounds = torch.from_numpy(lower_bounds).to(device).float(), torch.from_numpy(upper_bounds).to(device).float()
                outputs = lower_bounds + alt_sigmoid(model(inputs)) * (upper_bounds - lower_bounds)
            else:
                outputs = model(inputs)

            un_normalized_outputs = outputs * un_normalizer
            # print(inputs.shape, outputs.shape, un_normalizer.shape, un_normalized_outputs.shape)
            all_states.append(un_normalized_outputs.cpu().numpy()[0])

    print(all_states)
    all_states = np.array(all_states)
    print(all_states.shape)

    np.save(saveloc, all_states)
    print(f'saved to {saveloc}')
    return 

        



            

            


