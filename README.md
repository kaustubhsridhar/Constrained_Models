# Data and Models
The instructions to collect data and train the ARMAX constraint model for AP can be found at [README_for_Data_Collection.md](README_for_Data_Collection.md).

Alternatively, the processed data and trained models can be found inside each case study's directory at [this drive folder](https://drive.google.com/drive/folders/1L-aX46Xpkj7-1dps8lGuwRSAbENTX3lD?usp=sharing).

The raw data is also available inside each case study's directory at [this drive folder](https://drive.google.com/drive/folders/1mBGhZE1qdIXdwtYmAOgHUMsdHiW0YbxP?usp=sharing).

# Training Constrained Neural Network Dynamics Models
NAME_OF_ENV can be one of {Carla, Drones, AP, Quadrupeds}
```
bash run_Vanillas.sh
bash run_{NAME_OF_ENV}.sh
```

AP delta monotonicity analysis
```
bash run_delta_monotonicity.sh
```
