import random
import os
import numpy as np
import time
os.makedirs(f'../../../drones-datasets', exist_ok=True)

PHYSICS=f'pyb' # f'pyb_gnd_drag_dw'
TIME=30 # seconds

for N in [6]:
    for S in [1.0]:
        S = np.round(S, 3)
        for M in [1.0]:
            M = np.round(M, 3)
            print(N, S, M)
            os.system(f'nohup python -u hover_and_collect_data.py --duration_sec {TIME} --output_folder ../../../drones-datasets/{PHYSICS}_hover_{N}drones_{S}length_{M}mass --physics {PHYSICS} --num_drones {N} --length_scaling_factor {S} --mass_scaling_factor {M} &')
        
