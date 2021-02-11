# SIDM
Maintainer: Sean J. Wang, sjw2@andrew.cmu.edu

## Required Dependencies
[NumPy](numpy.org)
- Numpy is used for storage of parameters and small math operations

[PyTorch](pytorch.org)
- PyTorch is used for network training

[wheeledRobotSimPybullet](github.com/robomechanics/wheeledRobotSimPybullet)
- wheeledRobotSimPybullet is the simulation used to gather data. This package also contains the dataset class used for the replay buffer. It also contains state transformations used during training.

## Dataset
Dataset is generated using a PyBullet simulated wheeled robot. Two different systems are provided. The first has a novice and expert robot that are more similar to each other.

## Running SIDM Domain Transfer Learning
After the dataset is generated, SIDM training can be run using "sidmTraining.py"
