import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import pickle 
from torch.utils.data import Dataset, DataLoader

def get_dict(np_array, N_train, N_test):
    offset = 0
    # np.random.seed(0)
    # np.random.shuffle(np_array)
    return {'train': np_array[offset:N_train+offset, :], 'test': np_array[N_train+offset: N_train+offset + N_test, :]}

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.abs_max = -np.inf
        self.datapoint_of_abs_max = None

    def update(self, val, n=1, datapoint=None):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if abs(val) >= self.abs_max:
            self.abs_max = abs(val)
            self.datapoint_of_abs_max = datapoint

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

class QuadrupedDataset(Dataset):
    def __init__(self, data_dir=f'../GenLocoData', constraint_model_loc ='../data/networks', save_dir=f'../data', NumRobotsInEnv=1, robot_name = 'a1'):
        self.data_dir = data_dir
        self.dir = save_dir
        self.constraint_model_loc = constraint_model_loc
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(constraint_model_loc, exist_ok=True)
        if (not os.path.exists(f'{self.dir}/states.pkl')) or (not os.path.exists(f'{self.dir}/controls.pkl')) or (not os.path.exists(f'{self.dir}/next_states.pkl')) or (not os.path.exists(f'{self.dir}/traj_starts.pkl')):
            self.all_states, self.all_controls, self.all_next_states, self.all_traj_starts = {}, {}, {}, {}
            
            # main = pace
            observations_pace = np.load(f'{self.data_dir}/observations_{robot_name}_pace.npy')
            main_states = observations_pace[:-1, :12*15 + 6*15].copy()
            main_next_states = observations_pace[1:, :12].copy()
            main_controls = np.load(f'{self.data_dir}/actions_{robot_name}_pace.npy')[:-1, :]
            main_traj_starts = np.load(f'{self.data_dir}/episode_starts_{robot_name}_pace.npy')[:-1, :]

            # omega = spin
            observations_spin = np.load(f'{self.data_dir}/observations_{robot_name}_spin.npy')
            omega_states = observations_spin[:-1, :12*15 + 6*15].copy()
            omega_next_states = observations_spin[1:, :12].copy()
            omega_controls = np.load(f'{self.data_dir}/actions_{robot_name}_spin.npy')[:-1, :]
            omega_traj_starts = np.load(f'{self.data_dir}/episode_starts_{robot_name}_spin.npy')[:-1, :]

            N_train = 15000 
            N_test = 2000 

            N_train_omega = 15000 
            N_test_omega = 2000 

            all_states = np.concatenate((main_states, omega_states), axis=0)
            all_next_states = np.concatenate((main_next_states, omega_next_states), axis=0)
            all_controls = np.concatenate((main_controls, omega_controls), axis=0)
            x_normalizer = np.amax(np.abs(all_states), axis=0)
            next_x_normalizer = np.amax(np.abs(all_next_states), axis=0)
            u_normalizer = np.amax(np.abs(all_controls), axis=0)

            main_states /= x_normalizer; omega_states /= x_normalizer
            main_next_states /= next_x_normalizer; omega_next_states /= next_x_normalizer
            main_controls /= u_normalizer; omega_controls /= u_normalizer
            
            self.all_states['main'], self.all_controls['main'], self.all_next_states['main'], self.all_traj_starts['main'] = get_dict(main_states, N_train, N_test), get_dict(main_controls, N_train, N_test), get_dict(main_next_states, N_train, N_test), get_dict(main_traj_starts, N_train, N_test)
            self.all_states['omega'], self.all_controls['omega'], self.all_next_states['omega'], self.all_traj_starts['omega'] = get_dict(omega_states, N_train_omega, N_test_omega), get_dict(omega_controls, N_train_omega, N_test_omega), get_dict(omega_next_states, N_train_omega, N_test_omega), get_dict(omega_traj_starts, N_train_omega, N_test_omega)

            with open(f'{self.dir}/states.pkl', 'wb') as f:
                pickle.dump(self.all_states, f)
            with open(f'{self.dir}/controls.pkl', 'wb') as f:
                pickle.dump(self.all_controls, f)
            with open(f'{self.dir}/next_states.pkl', 'wb') as f:
                pickle.dump(self.all_next_states, f)
            with open(f'{self.dir}/traj_starts.pkl', 'wb') as f:
                pickle.dump(self.all_traj_starts, f)
            self.normalizers = {'x': x_normalizer, 'next_x': next_x_normalizer, 'u': u_normalizer}
            with open(f'{self.dir}/normalizers.pkl', 'wb') as f:
                pickle.dump(self.normalizers, f)
        else:
            with open(f'{self.dir}/states.pkl', 'rb') as f:
                self.all_states = pickle.load(f)
            with open(f'{self.dir}/controls.pkl', 'rb') as f:
                self.all_controls = pickle.load(f)
            with open(f'{self.dir}/next_states.pkl', 'rb') as f:
                self.all_next_states = pickle.load(f)
            with open(f'{self.dir}/traj_starts.pkl', 'rb') as f:
                self.all_traj_starts = pickle.load(f)
            with open(f'{self.dir}/normalizers.pkl', 'rb') as f:
                self.normalizers = pickle.load(f)

class InheritedDataset(QuadrupedDataset):
    def __init__(self, train=True):
        super(InheritedDataset, self).__init__()

        stage = 'train' if train else 'test'
        self.states = np.concatenate((self.all_states['main'][stage], self.all_states['omega'][stage]), axis=0)
        self.controls = np.concatenate((self.all_controls['main'][stage], self.all_controls['omega'][stage]), axis=0)
        self.next_states = np.concatenate((self.all_next_states['main'][stage], self.all_next_states['omega'][stage]), axis=0)

    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        state = self.states[idx]
        control = self.controls[idx]
        next_state = self.next_states[idx]

        x, y = np.concatenate((state, control)), next_state
        return (x, y)


def train(model, trainloader, device):
    criterion = nn.MSELoss()
    losses = AverageMeter()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

    num_epochs = 500
    for epoch in range(num_epochs):
        for idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device).float(), targets.to(device).float()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.shape[0])
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'{epoch=}, train {losses.avg=}')

    return model

def test(model, testloader, device):
    criterion = nn.MSELoss()
    losses = AverageMeter()
    optimizer = optim.Adam(model.parameters(), lr = 0.01)

    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device).float(), targets.to(device).float()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.shape[0])

    return losses.avg

if __name__ == "__main__":
    trainset, testset = InheritedDataset(train=True), InheritedDataset(train=False)
    batch_size = 64
    trainloader = DataLoader(trainset, batch_size, shuffle=False)
    testloader = DataLoader(testset, 1, shuffle=False)

    n_x = 12*15 + 6*15
    n_u = 12; n_next_x = 12
    model = prediction_model_quadruped(n_x + n_u, n_next_x)
    
    device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model = train(model, trainloader, device)
    print(f'test loss is {test(model, testloader, device)}')
    constraint_model_filename = f'{trainset.constraint_model_loc}/quadruped_all_net.pth'
    torch.save(model.state_dict(), constraint_model_filename)



