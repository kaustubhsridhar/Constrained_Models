import torch
import numpy as np
from torch.utils.data import Dataset
import json
from glob import glob
from torch.utils.data import DataLoader
import os
from dynamics_helpers import quad_dynamics, unicycle_dynamics, bicycle_dynamics, armax_constraint, spin_model_constraint
from neural_gas_helpers import data2memories, create_voronoi, voronoi_2_voronoi_bounds, voronoi_bounds_2_bounds
from scipy.spatial import Voronoi
import pickle 
from utils import get_samples, get_random_walk_samples, get_dict
from tqdm import tqdm
import time

class CARLADataset(Dataset):
    def __init__(self, data_dir=f'../CARLA/carla-datasets-Town01', save_dir=f'../CARLA/data', l_scales=[1.0], NumRobotsInEnv=1):
        self.data_dir = data_dir
        self.dir = save_dir
        os.makedirs(self.dir, exist_ok=True)

        self.dt = 0.1 # since gameTimestamp is in milliseconds in each json file # Proof of milliseconds: https://carla.readthedocs.io/en/stable/measurements/#:~:text=by%20the%20OS.-,game_timestamp,-uint32
        self.L = 2.3399999141693115 # bounding box extent in x diretcion is length of car # Proof: https://carla.readthedocs.io/en/stable/measurements/#:~:text=Transform%20and%20bounding%20box

        if (not os.path.exists(f'{self.dir}/states.pkl')) or (not os.path.exists(f'{self.dir}/controls.pkl')) or (not os.path.exists(f'{self.dir}/next_states.pkl')) or (not os.path.exists(f'{self.dir}/traj_starts.pkl')):
            self.all_states, self.all_controls, self.all_next_states, self.all_traj_starts = {}, {}, {}, {}

            episode_folders = glob(f'{self.data_dir}/episode_*')
            
            measurement_file_transitions = []
            for folder in episode_folders:
                files = glob(f'{folder}/measurements_*.json')
                transitions = [(files[i], files[i+1]) for i in range(len(files)-1)]
                measurement_file_transitions.append(transitions)

            main_states = []
            main_controls = []
            main_next_states = []
            main_traj_starts = []
            for trajectory in measurement_file_transitions:
                this_trajectory = []
                for (file_0, file_1) in trajectory:
                    with open(file_0) as f:
                        dict_0 = json.load(f)
                    with open(file_1) as f:
                        dict_1 = json.load(f)
                    
                    try:
                        state = np.array([dict_0["playerMeasurements"]["transform"]["location"]["x"], 
                                    dict_0["playerMeasurements"]["transform"]["location"]["y"], 
                                    dict_0["playerMeasurements"]["transform"]["rotation"]["yaw"] * np.pi/180.0,
                                    # dict_0["playerMeasurements"]["autopilotControl"]["steer"] * 70 * np.pi/180.0, # See https://carla.readthedocs.io/en/stable/measurements/#:~:text=carla_client.send_control(control)-,(*),-The%20actual%20steering
                                    ])
                        next_state = np.array([dict_1["playerMeasurements"]["transform"]["location"]["x"], 
                                    dict_1["playerMeasurements"]["transform"]["location"]["y"], 
                                    dict_1["playerMeasurements"]["transform"]["rotation"]["yaw"] * np.pi/180.0,
                                    # dict_1["playerMeasurements"]["autopilotControl"]["steer"] * 70 * np.pi/180.0, # See https://carla.readthedocs.io/en/stable/measurements/#:~:text=carla_client.send_control(control)-,(*),-The%20actual%20steering                                    
                                    ])
                        # steering_rate = (next_state[3] - state[3])/self.dt
                        angular_rate = (next_state[2] - state[2])/self.dt
                        control = np.array([
                                        dict_0["playerMeasurements"]["forwardSpeed"],
                                        angular_rate # steering_rate, 
                                        ])
                        if control[0] >= 0.5:
                            if len(this_trajectory) == 0:
                                this_trajectory.append([1])
                            else:
                                this_trajectory.append([0])

                            main_states.append(state)
                            main_controls.append(control)
                            main_next_states.append(next_state)
                        else:
                            main_traj_starts.extend(this_trajectory)
                            this_trajectory = []
                            continue
                    except:
                        main_traj_starts.extend(this_trajectory)
                        this_trajectory = []
                        continue
                main_traj_starts.extend(this_trajectory)

            main_states, main_controls, main_next_states, main_traj_starts = np.array(main_states), np.array(main_controls), np.array(main_next_states), np.array(main_traj_starts)
            
            x_normalizer = np.amax(np.abs(main_states), axis=0)
            u_normalizer = np.amax(np.abs(main_controls), axis=0)

            main_states = main_states / x_normalizer
            main_controls = main_controls / u_normalizer
            main_next_states = main_next_states / x_normalizer

            lower_state_bounds = np.amin(main_states, axis=0) / x_normalizer
            upper_state_bounds = np.amax(main_states, axis=0) / x_normalizer
            
            omega_states = get_samples(num_samples=len(main_states), lower_bounds=lower_state_bounds, upper_bounds=upper_state_bounds) # 
            omega_traj_starts = np.ones((len(omega_states), 1))
            # omega_states, omega_traj_starts = get_random_walk_samples(num_samples=len(main_states), lower_bounds=lower_state_bounds, upper_bounds=upper_state_bounds, indices=[0, 1], step_size=0.01, max_traj_steps = 100)
            
            omega_controls = np.zeros((len(omega_states), 2)) # get_samples(num_samples=len(main_states), lower_bounds=[0]*2, upper_bounds=[0]*2)
            omega_next_states = omega_states.copy()

            N_train = 15000 # 15000 (OG), 6000 (6k6kdata), 10000 (10k5kdata)
            N_test = 1500 

            N_train_omega = 15000 # 5000 (OG), 6000 (6k6kdata), 5000 (10k5kdata)
            N_test_omega = 1500 
            
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
            self.normalizers = {'x': x_normalizer, 'u': u_normalizer}
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

        self.x_normalizer = self.normalizers['x']
        self.u_normalizer = self.normalizers['u']
        self.family_of_dynamics = []
        for l in l_scales:
            # self.family_of_dynamics.append(bicycle_dynamics(L=l * self.L, dt=self.dt, x_normalizer=x_normalizer, u_normalizer=u_normalizer))
            self.family_of_dynamics.append(unicycle_dynamics(L=l * self.L, dt=self.dt, x_normalizer=self.x_normalizer, u_normalizer=self.u_normalizer))

        self.output_un_normalizer = self.normalizers['x']

class DronesDataset(Dataset):
    """
    States: (x, y, z, q0, q1, q2, q3, r, p, y, vx, vy, vz, wr, wp, wy, rpm1, rpm2, rpm3, rpm4) ~ 20
    Control: (tgt_x, tgt_y, tgt_z, tgt_r, tgt_p, tgt_y, tgt_rpm1, tgt_rpm2, tgt_rpm3, tgt_rpm4) ~ 10 but we use only last 4
    Next_states: (x, y, z, r, p, y, vx, vy, vz, wr, wp, wy) ~ 12 [See ./dynamics_helpers.py --> quad_step()]
    """
    def __init__(self, NumRobotsInEnv=6, data_dir=f'../Drones/drones-datasets', save_dir = f'../Drones/data', m_scales=[1.0], l_scales=[1.0]): # m_scales=np.arange(0.8, 1.25, 0.1)
        self.data_dir = data_dir
        self.dir = save_dir
        os.makedirs(self.dir, exist_ok=True)

        if (not os.path.exists(f'{self.dir}/states.pkl')) or (not os.path.exists(f'{self.dir}/controls.pkl')) or (not os.path.exists(f'{self.dir}/next_states.pkl')) or (not os.path.exists(f'{self.dir}/traj_starts.pkl')):
            self.all_states, self.all_controls, self.all_next_states, self.all_traj_starts = {}, {}, {}, {}

            self.NumRobotsInEnv = NumRobotsInEnv # can also be '*' for all numbers of drones!
        
            PHYSICS=f'pyb_gnd_drag_dw' # f'pyb' (OG), f'pyb_gnd_drag_dw' (GndDragDwData)
            main_folders = glob(f'{self.data_dir}/{PHYSICS}_{self.NumRobotsInEnv}drones_1.0length_1.0mass')
            self.all_states['main'], self.all_controls['main'], self.all_next_states['main'], self.all_traj_starts['main'] = self.load_from_folders(main_folders, data_type='main')
            
            hover_folders = glob(f'{self.data_dir}/{PHYSICS}_hover_{self.NumRobotsInEnv}drones_1.0length_1.0mass')
            hover_states, hover_controls, hover_next_states, hover_traj_starts = self.load_from_folders(hover_folders, data_type='hover')
            self.all_states['omega'], self.all_controls['omega'], self.all_next_states['omega'], self.all_traj_starts['omega'] = hover_states, hover_controls, hover_next_states, hover_traj_starts

            with open(f'{self.dir}/states.pkl', 'wb') as f:
                pickle.dump(self.all_states, f)
            with open(f'{self.dir}/controls.pkl', 'wb') as f:
                pickle.dump(self.all_controls, f)
            with open(f'{self.dir}/next_states.pkl', 'wb') as f:
                pickle.dump(self.all_next_states, f)
            with open(f'{self.dir}/traj_starts.pkl', 'wb') as f:
                pickle.dump(self.all_traj_starts, f)
        else:
            with open(f'{self.dir}/states.pkl', 'rb') as f:
                self.all_states = pickle.load(f)
            with open(f'{self.dir}/controls.pkl', 'rb') as f:
                self.all_controls = pickle.load(f)
            with open(f'{self.dir}/next_states.pkl', 'rb') as f:
                self.all_next_states = pickle.load(f)
            with open(f'{self.dir}/traj_starts.pkl', 'rb') as f:
                self.all_traj_starts = pickle.load(f)

        self.family_of_dynamics = []
        for m in m_scales:
            for l in l_scales:
                self.family_of_dynamics.append(quad_dynamics(m, l))

        self.output_un_normalizer = np.array([1])

    def load_from_folders(self, folders, data_type):
        if data_type == 'hover':
            N_train = 15000 # OG: 6000
            N_test = 2000 # OG: 1200
        else:
            N_train = 15000 # OG: 6000
            N_test = 2000 # OG: 1200
        end1 = int((N_train+N_test)/(self.NumRobotsInEnv) * 1/2)
        end2 = int((N_train+N_test)/(self.NumRobotsInEnv) * 2/2)
        # end3 = int((N_train+N_test)/(self.NumRobotsInEnv) * 3/4)
        # end4 = int((N_train+N_test)/(self.NumRobotsInEnv) * 1) + 100
            
        state_npys = []
        control_npys = []
        next_state_npys = []
        traj_start_indicators = []
        for folder in folders:
            for npy_file_loc in glob(f'{folder}/*.npy'):
                npy_file = np.load(npy_file_loc)
                num_quadrotors = npy_file['states'].shape[0]
                
                for (start, end) in [(0, end1), (end1, end2)]:
                    for i in range(num_quadrotors):
                        state_npys.append(npy_file['states'][i, :, start:end].T) # npy_file['state'] ~ (num_drones, num_states=20, num_datapoints) 
                        xyz = npy_file['states'][i, :3, start+1:end+1]
                        rpy_vxvyvz_wrwpwy = npy_file['states'][i, 7:16, start+1:end+1]
                        combined = np.concatenate((xyz, rpy_vxvyvz_wrwpwy), axis=0)                        
                        next_state_npys.append(combined.T)
                        control_npys.append(npy_file['controls'][i, -4:, start:end].T) # last four elements are four motor RPMs
                        
                        num_datapoints = end-start
                        traj_start_indicators.extend( [[1]] + [[0]]*(num_datapoints-1))
        
        states = np.concatenate(state_npys, axis=0)
        controls = np.concatenate(control_npys, axis=0)
        next_states = np.concatenate(next_state_npys, axis=0)
        traj_start_indicators = np.array(traj_start_indicators)

        # Normalizing all angular velocities
        states[:, -4:] /= 10000
        controls /= 10000

        return get_dict(states, N_train, N_test), get_dict(controls, N_train, N_test), get_dict(next_states, N_train, N_test), get_dict(traj_start_indicators, N_train, N_test)


class QuadrupedDataset(Dataset):
    def __init__(self, data_dir=f'../Quadrupeds/GenLocoData', constraint_model_loc ='../Quadrupeds/data/networks', save_dir=f'../Quadrupeds/data', NumRobotsInEnv=1, robot_name = 'a1'):
        import sys
        sys.path.append('../')
        from Quadrupeds.train_constraint_model.main import prediction_model_quadruped as constraint_model_class
        
        self.data_dir = data_dir
        self.dir = save_dir
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(constraint_model_loc, exist_ok=True)
        if (not os.path.exists(f'{self.dir}/states.pkl')) or (not os.path.exists(f'{self.dir}/controls.pkl')) or (not os.path.exists(f'{self.dir}/next_states.pkl')) or (not os.path.exists(f'{self.dir}/traj_starts.pkl')):
            self.all_states, self.all_controls, self.all_next_states, self.all_traj_starts = {}, {}, {}, {}
            
            # main = pace
            main_gait = 'spin' # OG: 'pace'
            observations_main = np.load(f'{self.data_dir}/observations_{robot_name}_{main_gait}.npy')
            main_states = observations_main[:-1, :12*15 + 6*15].copy()
            main_next_states = observations_main[1:, :12].copy()
            main_controls = np.load(f'{self.data_dir}/actions_{robot_name}_{main_gait}.npy')[:-1, :]
            main_traj_starts = np.load(f'{self.data_dir}/episode_starts_{robot_name}_{main_gait}.npy')[:-1, :]

            # omega = spin
            omega_gait = 'pace' # OG: 'spin'
            observations_omega = np.load(f'{self.data_dir}/observations_{robot_name}_{omega_gait}.npy')
            omega_states = observations_omega[:-1, :12*15 + 6*15].copy()
            omega_next_states = observations_omega[1:, :12].copy()
            omega_controls = np.load(f'{self.data_dir}/actions_{robot_name}_{omega_gait}.npy')[:-1, :]
            omega_traj_starts = np.load(f'{self.data_dir}/episode_starts_{robot_name}_{omega_gait}.npy')[:-1, :]

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

        # overwrite trajectory starts
        N_train = 15000 
        N_test = 2000 
        N_train_omega = 15000 
        N_test_omega = 2000 
        self.all_traj_starts['main']['train'] = np.ones((N_train, 1))
        self.all_traj_starts['main']['test'] = np.ones((N_test, 1))
        self.all_traj_starts['omega']['train'] = np.ones((N_train_omega, 1))
        self.all_traj_starts['omega']['test'] = np.ones((N_test_omega, 1))
        
        n_x = 12*15 + 6*15
        n_u = 12; n_next_x = 12
        model = constraint_model_class(n_x + n_u, n_next_x)
        constraint_model_filename = f'{constraint_model_loc}/quadruped_all_net.pth'
        model.load_state_dict(torch.load(constraint_model_filename, map_location=torch.device("cpu")))
        self.family_of_dynamics = [spin_model_constraint(model, self.normalizers['x'], self.normalizers['next_x'], self.normalizers['u'])]

        self.output_un_normalizer = np.array([self.normalizers['next_x']])
        # self.input_un_normalizer = np.array([self.normalizers['x']] + [self.normalizers['u']])

class APDataset(Dataset):
    def __init__(self, data_dir=f'../AP/insulin_matlab', constraint_model_loc ='../AP/data/networks', save_dir=f'../AP/data', NumRobotsInEnv=1):
        import sys
        sys.path.append(f'../')
        from AP.data_manager import create_regression_data_with_meals_reformatted
        from AP.armax_model_on_reformatted_data import armax_model, fit_model, test_model

        self.data_dir = data_dir
        self.dir = save_dir
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(constraint_model_loc, exist_ok=True)
        if (not os.path.exists(f'{self.dir}/states.pkl')) or (not os.path.exists(f'{self.dir}/controls.pkl')) or (not os.path.exists(f'{self.dir}/next_states.pkl')) or (not os.path.exists(f'{self.dir}/traj_starts.pkl')):
            self.all_states, self.all_controls, self.all_next_states, self.all_traj_starts = {}, {}, {}, {}
            
            glucose_src = f"{self.data_dir}/Glucose_data_above150.csv"
            insulin_src = f"{self.data_dir}/Insulin_data_above150.csv"
            meals_src = f"{self.data_dir}/meal_data_above150.csv"
            main_states, main_controls, main_next_states, main_traj_starts, glucose_normalizer, insulin_normalizer, meal_normalizer = create_regression_data_with_meals_reformatted(glucose_src, insulin_src, meals_src)
            
            glucose_src = f"{self.data_dir}/Glucose_data_zeroCarbs.csv"
            insulin_src = f"{self.data_dir}/Insulin_data_zeroCarbs.csv"
            meals_src = f"{self.data_dir}/meal_data_zeroCarbs.csv"
            omega_states, omega_controls, omega_next_states, omega_traj_starts, g_n, i_n, m_n = create_regression_data_with_meals_reformatted(glucose_src, insulin_src, meals_src, glucose_normalizer, insulin_normalizer, meal_normalizer)
            
            assert (glucose_normalizer == g_n) and (insulin_normalizer == i_n) and (meal_normalizer == m_n)

            print(f'{main_states.shape=}, {main_controls.shape=}, {main_next_states.shape=}, {main_traj_starts.shape=}')
            print(f'{omega_states.shape=}, {omega_controls.shape=}, {omega_next_states.shape=}, {omega_traj_starts.shape=}')

            N_train = 15750 
            N_test = 2000 

            N_train_omega = 15750 
            N_test_omega = 2000 
            
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
            self.normalizers = {'glucose': glucose_normalizer, 'insulin': insulin_normalizer, 'meal': meal_normalizer}
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

        model = armax_model()
        constraint_model_filename = f'{constraint_model_loc}/diabetes_armax_model_state_dict.pth'
        if (not os.path.exists(constraint_model_filename)):
            print(f'Training constraint model (ARMAX model).....')
            model = fit_model(self.all_states, self.all_controls, self.all_next_states, self.normalizers['glucose'], self.normalizers['insulin'], self.normalizers['meal'], filename=constraint_model_filename)
            model.to(torch.device("cpu"))
        else:
            model.load_state_dict(torch.load(constraint_model_filename, map_location=torch.device("cpu")))
        self.family_of_dynamics = [armax_constraint(model, self.normalizers['glucose'], self.normalizers['insulin'], self.normalizers['meal'])]

        self.output_un_normalizer = np.array([self.normalizers['glucose']])
        self.input_un_normalizer = np.array([self.normalizers['glucose']]*10 + [self.normalizers['insulin']]*10 + [self.normalizers['meal']]*10 +[self.normalizers['insulin']])

def make_BoundsAndDynamicsDataset_instance(env, max_memories=1000, gng_epochs=1, NumRobotsInEnv=6, num_voronoi_samples=25, delta=0.05, DELETE=False, seed=0):
    """
    Returns a BoundsAndDynamicsDataset instance that inherits from ...
    ... the BoundsDataset class that itself inherits from the cls corresponding to env argument.
    """
    if seed == 0 or env == 'AP':
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        seed = 0 # overwrite to reload memories and bounds

    if env == "Drones":
        base_class = DronesDataset
    elif env == "Carla":
        base_class = CARLADataset
    elif env == "Quadrupeds":
        base_class = QuadrupedDataset
    elif env == "AP":
        base_class = APDataset

    class BoundsAndDynamicsDataset(base_class):
        def __init__(self):
            super(BoundsAndDynamicsDataset, self).__init__(NumRobotsInEnv=NumRobotsInEnv)
            
            extra_env_info = f'_{NumRobotsInEnv}drones' if env == 'Drones' else f''
            extra_info = f'_{gng_epochs}gngepochs' if gng_epochs > 1 else f''
            extra_info += f'' if delta == 0.05 else f'_{delta}delta'
            seed_info = f'' if seed == 0 else f'_seed{seed}'

            gng_path = f'{self.dir}/gng{extra_env_info}_{max_memories}memories{extra_info}{seed_info}.pkl'
            voronoi_path = f'{self.dir}/voronoi{extra_env_info}_{max_memories}memories{extra_info}{seed_info}.pkl'
            voronoi_bounds_path = f'{self.dir}/voronoiBounds{extra_env_info}_{max_memories}memories{extra_info}{seed_info}.pkl'
            bounds_path = f'{self.dir}/bounds{extra_env_info}_{max_memories}memories{extra_info}{seed_info}.pkl'
            dynamics_next_states_path = f'{self.dir}/dynamics_next_states{extra_env_info}.pkl' # {seed_info}.pkl'
            self.all_saved_pkl_paths = [gng_path, voronoi_path, voronoi_bounds_path, bounds_path]

            for obj in ['self.all_states', 'self.all_controls', 'self.all_next_states', 'self.all_traj_starts', 'self.family_of_dynamics']:
                assert eval(obj), f"{obj} does not exist!"

            # for data_type in ['main', 'omega']:
            #     for stage in ['train', 'test']:   
            #         print(f'\n----------{data_type=} and {stage=}')
            #         print(f"{self.all_states[data_type][stage].shape=}")
            #         print(f"{self.all_controls[data_type][stage].shape=}")
            #         print(f"{self.all_next_states[data_type][stage].shape=}")
            #         print(f"{self.all_traj_starts[data_type][stage].shape=}")

            # For below and neural network setup outside
            self.num_control_inputs = self.all_controls['main']['train'].shape[1]
            self.input_size = self.all_states['main']['train'].shape[1] + self.num_control_inputs
            self.output_size = self.all_next_states['main']['train'].shape[1]

            # Collect only training data for below 
            self.all_train_states = np.concatenate((self.all_states['main']['train'], self.all_states['omega']['train']), axis=0)
            self.all_train_controls = np.concatenate((self.all_controls['main']['train'], self.all_controls['omega']['train']), axis=0)
            self.all_train_traj_starts = np.concatenate((self.all_traj_starts['main']['train'], self.all_traj_starts['omega']['train']), axis=0)
            
            # Memories
            if (not os.path.exists(gng_path)):
                self.gng = data2memories(self.all_train_states, self.all_train_controls, max_memories=max_memories, gng_epochs=gng_epochs)
                with open(gng_path, 'wb') as f:
                    pkl = pickle.Pickler(f, protocol=4)
                    pkl.dump(self.gng)
            else:
                with open(gng_path, 'rb') as f:
                    unpkl = pickle.Unpickler(f)
                    self.gng = unpkl.load()
            print(f'num of nodes = {len(self.gng.graph.nodes)} and num of edges = {len(list(self.gng.graph.edges.keys()))}')

            # Voronoi
            if (not os.path.exists(voronoi_path)):
                voronoi = create_voronoi(self.gng)
                with open(voronoi_path, 'wb') as f:
                    pickle.dump(voronoi, f)
            else:
                with open(voronoi_path, 'rb') as f:
                    voronoi = pickle.load(f)
            self.midpoints, self.normals, self.offsets = voronoi


            # Bounds
            if (not os.path.exists(bounds_path)):  
                self.voronoi_bounds, _, _ = voronoi_2_voronoi_bounds(self.gng, self.midpoints, self.all_train_states, self.all_train_controls, self.all_train_traj_starts, self.family_of_dynamics, self.num_control_inputs, num_samples=num_voronoi_samples)

                self.all_bounds = {}
                for data_type in ['main', 'omega']:
                    self.all_bounds[data_type] = {}
                    for stage in ['train', 'test']:   
                        print(f'Getting bounds of {data_type=} {stage=}:')        
                        
                        # self.voronoi_bounds, _, _ = voronoi_2_voronoi_bounds(self.gng, self.midpoints, self.all_states[data_type][stage], self.all_controls[data_type][stage], self.all_traj_starts[data_type][stage], self.family_of_dynamics, self.num_control_inputs, num_samples=num_voronoi_samples)

                        self.all_bounds[data_type][stage] = voronoi_bounds_2_bounds(self.gng, self.midpoints, self.voronoi_bounds, self.all_states[data_type][stage], self.all_controls[data_type][stage], self.all_next_states[data_type][stage], self.all_traj_starts[data_type][stage], delta=delta)

                with open(voronoi_bounds_path, 'wb') as f:
                    pickle.dump(self.voronoi_bounds, f)
                with open(bounds_path, 'wb') as f:
                    pickle.dump(self.all_bounds, f)
            else:
                with open(voronoi_bounds_path, 'rb') as f:
                    self.voronoi_bounds = pickle.load(f)
                with open(bounds_path, 'rb') as f:
                    self.all_bounds = pickle.load(f)

            # Dyanmics next states
            if (not os.path.exists(dynamics_next_states_path)):
                self.all_dynamics_next_states = {}
                for data_type in ['main', 'omega']:
                    self.all_dynamics_next_states[data_type] = {}
                    for stage in ['train', 'test']:   
                        print(f'Getting dynamics_next_states of {data_type=} {stage=}:') 

                        next_states = []
                        for dyn_idx, dyn in enumerate(self.family_of_dynamics):
                            next_states.append([])
                            for data_idx, (state, control) in enumerate(zip(self.all_states[data_type][stage], self.all_controls[data_type][stage])):
                                next_states[dyn_idx].append(dyn([state, control]))

                        self.all_dynamics_next_states[data_type][stage] = np.array(next_states) # (num_dynamics, num_datapoints, 12)

                with open(dynamics_next_states_path, 'wb') as f:
                    pickle.dump(self.all_dynamics_next_states, f)
            else:
                with open(dynamics_next_states_path, 'rb') as f:
                    self.all_dynamics_next_states = pickle.load(f)

    class ConstrainedDataset(BoundsAndDynamicsDataset):
        def __init__(self, data_type='main', train=True):
            super(ConstrainedDataset, self).__init__()

            stage = 'train' if train else 'test'
            self.states = self.all_states[data_type][stage]
            self.controls = self.all_controls[data_type][stage]
            self.next_states = self.all_next_states[data_type][stage]
            self.traj_starts = self.all_traj_starts[data_type][stage]
            self.lower_bounds = self.all_bounds[data_type][stage]['lo']
            self.upper_bounds = self.all_bounds[data_type][stage]['up']
            self.dynamics_next_states = self.all_dynamics_next_states[data_type][stage]

        def __len__(self):
            return len(self.states)
            
        def __getitem__(self, idx):
            state = self.states[idx]
            control = self.controls[idx]
            next_state = self.next_states[idx]
            is_traj_starting = self.traj_starts[idx]
            upper_bound = self.upper_bounds[idx, :]
            lower_bound = self.lower_bounds[idx, :]
            list_of_dynamics_next_states = self.dynamics_next_states[:, idx, :]

            x, y = np.concatenate((state, control)), next_state
            return (x, y, is_traj_starting, lower_bound, upper_bound, list_of_dynamics_next_states)
                

    objs = (ConstrainedDataset(data_type='main', train=True), ConstrainedDataset(data_type='main', train=False), 
            ConstrainedDataset(data_type='omega', train=True), ConstrainedDataset(data_type='omega', train=False))

    if DELETE:
        for path in objs[0].all_saved_pkl_paths:
            if os.path.exists(path):
                os.remove(path)

    return objs


def make_LagrangianDataset_instance(env, NumRobotsInEnv=6, seed=0):
    """
    For Vanilla / Lagrangian
    """
    if seed == 0 or env == 'AP':
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        seed = 0 # overwrite to reload memories and bounds

    if env == "Drones":
        base_class = DronesDataset
    elif env == "Carla":
        base_class = CARLADataset
    elif env == "Quadrupeds":
        base_class = QuadrupedDataset
    elif env == "AP":
        base_class = APDataset

    class OnlyDynamicsDataset(base_class):
        def __init__(self):
            super(OnlyDynamicsDataset, self).__init__(NumRobotsInEnv=NumRobotsInEnv)

            extra_env_info = f'_{NumRobotsInEnv}drones' if env == 'Drones' else f''
            seed_info = f'' if seed == 0 else f'_seed{seed}'
            dynamics_next_states_path = f'{self.dir}/dynamics_next_states{extra_env_info}.pkl' # {seed_info}.pkl'
            
            for obj in ['self.all_states', 'self.all_controls', 'self.all_next_states', 'self.all_traj_starts', 'self.family_of_dynamics']:
                assert eval(obj), f"{obj} does not exist!"
            
            # For neural network setup outside
            self.num_control_inputs = self.all_controls['main']['train'].shape[1]
            self.input_size = self.all_states['main']['train'].shape[1] + self.num_control_inputs
            self.output_size = self.all_next_states['main']['train'].shape[1]

            # Dyanmics next states
            if (not os.path.exists(dynamics_next_states_path)):
                self.all_dynamics_next_states = {}
                for data_type in ['main', 'omega']:
                    self.all_dynamics_next_states[data_type] = {}
                    for stage in ['train', 'test']:   
                        print(f'Getting dynamics_next_states of {data_type=} {stage=}:') 

                        next_states = []
                        for dyn_idx, dyn in enumerate(self.family_of_dynamics):
                            next_states.append([])
                            for data_idx, (state, control) in enumerate(zip(self.all_states[data_type][stage], self.all_controls[data_type][stage])):
                                next_states[dyn_idx].append(dyn([state, control]))

                        self.all_dynamics_next_states[data_type][stage] = np.array(next_states) # (num_dynamics, num_datapoints, 12)

                with open(dynamics_next_states_path, 'wb') as f:
                    pickle.dump(self.all_dynamics_next_states, f)
            else:
                with open(dynamics_next_states_path, 'rb') as f:
                    self.all_dynamics_next_states = pickle.load(f)
            
    class LagrangianDataset(OnlyDynamicsDataset):
        def __init__(self, data_type='main', train=True):
            super(LagrangianDataset, self).__init__()

            stage = 'train' if train else 'test'
            self.states = self.all_states[data_type][stage]
            self.controls = self.all_controls[data_type][stage]
            self.next_states = self.all_next_states[data_type][stage]
            self.traj_starts = self.all_traj_starts[data_type][stage]
            self.dynamics_next_states = self.all_dynamics_next_states[data_type][stage]

        def __len__(self):
            return len(self.states)
            
        def __getitem__(self, idx):
            state = self.states[idx]
            control = self.controls[idx]
            next_state = self.next_states[idx]
            is_traj_starting = self.traj_starts[idx]
            upper_bound = -1
            lower_bound = -1
            list_of_dynamics_next_states = self.dynamics_next_states[:, idx, :]

            x, y = np.concatenate((state, control)), next_state
            return (x, y, is_traj_starting, lower_bound, upper_bound, list_of_dynamics_next_states)

    return (LagrangianDataset(data_type='main', train=True), LagrangianDataset(data_type='main', train=False), 
            LagrangianDataset(data_type='omega', train=True), LagrangianDataset(data_type='omega', train=False))


if __name__ == "__main__":

    dataset = DronesDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    for idx, (states, targets) in enumerate(dataloader):
        print(states.shape, targets.shape, '\n\n\n')
        break
