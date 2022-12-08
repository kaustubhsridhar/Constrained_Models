# Data Collection
## Car (CARLA)
### Conda Environment
```
cd CARLA/data_collector
conda env create -f requirements.yml
conda activate carla_data
```

### Start CarlaGear server (first terminal)
Donwload the server from here: https://drive.google.com/file/d/1X52PXqT0phEi5WEWAISAQYZs-Ivx4VoE/view
```
cd CARLA/CarlaGear
sh CarlaUE4.sh /Game/Maps/Town01 -windowed -world-port=2000  -benchmark -fps=10
```

### Data Collection Code (second terminal)
```
cd ../data-collector/
python -u collect.py --data-path ../carla-datasets-Town01
```

## Drones (GYM-PYBULLET-DRONES)
### Conda Environment
```
conda create -n drones python=3.8
conda activate drones
cd Drones/gym-pybullet-drones/
pip install -e .
```

### Data Collection Code
N Crazyflie X drones of different length_scaling_factors and mass_scaling_factors in circular flight and hover.
```
cd Drones/gym-pybullet-drones/gym_pybullet_drones/examples
python collect_data.py
python collect_hover_data.py
```

## AP 
Automatic (but for more info please see AP/insulin_matlab/Readme.md)

## Quadrupeds (PyBullet)
### Conda Environment
```
conda create -n Quadrupeds python=3.7
conda activate Quadrupeds
cd Quadrupeds/GenLoco
python setup.py install --user
sudo apt install libopenmpi-dev
pip install -r requirements.txt
```

### Data Collection Code
```
cd Quadrupeds/GenLoco
bash save_data.sh
```

# Training Constrained model for AP
```
cd AP/
```
Adjust mode in main.py and then run.
```
python main.py
```
