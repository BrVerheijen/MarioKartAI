# Mario Kart AI
This project contains an integration for Super Mario Kart for the Snes to be used as a gym environment with gym retro.

We have created a number of different reward shapings to experiment the capabilities of the AI. They AI trains for a specific circuit and does not learn the drive on other circuits.

## Requirements
-Python version: 3.8.10

## Setup
1. Install gym-retro version 0.8.0 with ```pip install gym-retro=0.8.0```
2. Install stable-baselines3 version 1.4.0 with ```pip install stable-baselines3```
3. Install Tensorflow and Tensorboard version 2.8.0 with ```pip install Tensorflow=2.8.0``` and ```pip install Tensorboard=2.8.0```
4. Move the ```SuperMarioKart-Snes``` folder to the ```data/stable/``` folder within your gym-retro installation

## Setup GPU integration
1. Install CUDA with the installation guides found here
Windows: ```https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html```
Linux: ```https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html```
2. Install Pytorch with CUDA integration here ```https://pytorch.org/get-started/locally/```

## Training
To train the agent you should run.
```python main.py```
-- To Check the learned model
```python test_model_multiple.py --load best_model.zip```

## Contributors
-DGraus: Dajmen Graus
-LorenzoClermonts: Lorenzo Clermonts
-BrVerheijen: Bram Verheijen
-BasRuyters: Bas Ruijters
