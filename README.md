# Accelerated composite reliability assessment using a reinforcement learning-driven graph neural network surrogate

**Repository accompanying:**

"Accelerated composite reliability assessment using a reinforcement learning-driven graph neural network surrogate" by Fernando A. Assis, Marcos Netto, and Arnob Ghosh

---

## Overview
This repository provides the Python implementation of the framework proposed in our paper, which accelerates power system reliability assessment by combining non-sequential Monte Carlo simulation (NS-MCS) with a reinforcement learning-driven graph neural network (RL-GNN) surrogate model to speed up AC-OPF evaluations. It is intended for reproducibility and extension by other researchers.


## Prerequisites & Installation

1. **Requirements:**
- Python 3.8+
- PyTorch
- PyTorch Geometric
- Gymnasium
- PYPOWER
- NumPy
- SciPy

2. **Quick Install:**
```pip install torch torch-geometric numpy scipy pypower gymnasium```

3. **Clone the repository**
Clone the repository:
```bash
git clone https://github.com/FAssisRepository/ReliabilityAssessment_RL.git
cd ReliabilityAssessment_RL
```

## Repository structure

```bash
.
├── main.py                       # Main execution script running the full training and evaluation pipeline
├── InputData/                    # Directory with system datasets, load curve, and central configuration file
│   ├── 00_SIMULATION_SETTINGS.txt    # Central configuration file (System selection and NS-MCS, GNN, and training settings)
│   ├── IEEERTS_LOAD.load             # Full-year hourly load profile curve
│   ├── SIST_6BUS_GLOBAL_AC.dat       # RBTS 6-bus system data
│   ├── IEEERTS79_GLOBAL_AC.dat       # IEEE-RTS 24-bus system data
│   └── SIST_200B_GLOBAL_AC.dat       # Illinois 200-bus (ACTIVSg200) system data
├── GNNmodel/                     # Directory for saving trained GNN model checkpoints
│   └── .gitkeep
├── OutputData/                   # Directory for storing simulation results and log files
│   └── .gitkeep
├── input_data_class.py           # Data parser loading system topologies, network parameters, and settings
├── auxiliar_classes.py           # Helper utilities and data structures for simulation logging and tracking
├── DRL_reliab_env_1episode.py    # Single-episode RL environment handling graph states, actions, rewards, and constraints
├── element_classes.py            # Object models for power system elements (buses, generators, circuits, loads)
├── flow_AC.py                    # AC Power Flow (AC-PF) solver (without optimization)
├── OPF_AC.py                     # AC Optimal Power Flow (AC-OPF) solver
└── reliab_assessment.py          # Primary reliability assessment engine, GNN training, and evaluation tests
│
└── test_systems_scenarios_description.pdf   # Detailed description of the test systems and scenario generation
```

## Configuration

Simulation and training parameters are primarily defined in:
```
InputData/00_SIMULATION_SETTINGS.txt
```
This file contains the main configuration parameters for:
- Selection of the test system;
- NS-MCS settings;
- Load profile and simulation settings;
- GNN configuration;
- Supervised learning parameters;
- Reinforcement learning parameters;
- Training and evaluation settings.

Before running an experiment, review this file and adjust the parameters according to the desired test case.

For a detailed description of the test systems and scenario-generation procedures, see:
```
test_systems_scenarios_description.pdf
```

## Running the Framework

The main execution script is:
```
main.py
```
The framework performs the corresponding training and reliability assessment procedures according to the settings specified in:
```
InputData/00_SIMULATION_SETTINGS.txt
```
Simulation results, log files, and trained model checkpoints are stored in the corresponding output directories in:
```
OutputData/
```

## Reproducibility
To reproduce the experiments reported in the accompanying paper:
- Install the required Python dependencies.
- Clone the repository.
- Select the desired test system and simulation parameters in ```InputData/00_SIMULATION_SETTINGS.txt```.
- Run ```main.py```.
- Use the generated output files in the ```OutputData/``` directory to check the results, including training dynamics, corrective performance metrics, and reliability indices.
	
The main output files associated with the results reported in the paper are described below:
- **RL training dynamics**: Results for the per-constraint cost signals and Lagrange multipliers are available in files following the naming convention ```00_log_Lagrangian_Test_x.txt```, where ```x``` corresponds to the test number defined in ```InputData/00_SIMULATION_SETTINGS.txt```.
- **Corrective performance metrics**: Optimality and feasibility results for the 1,000 unseen evaluation states are also available in files following the naming convention ```00_log_Lagrangian_Test_x.txt```.
- **Corrective controls**: Corrective control actions for each evaluation state can be found in files following the naming conventions: 
	- ```AC-OPF-OPF-y.txt``` - AC-OPF benchmark solution obtained using a conventional interior-point solver; 
	- ```AC-PF-GNN-PT-y.txt``` - the corrective control and complete power flow solution obtained using the SL-GNN policy; 
	- ```AC-PF-GNN-RL-y.txt``` - the corrective control and complete power flow solution obtained using the RL-GNN policy.
Here, ```y``` corresponds to the evaluation-state number.
- **Reliability assessment results**: The results corresponding to Table VI in the paper are available in files following the naming conventions: 
	- ```x_z_OPF_NS_MCS.txt``` - AC-OPF-NS-MCS benchmark;
	- ```x_z_RL_GNN_NS_MCS.txt``` - RL-GNN-NS-MCS;
	- ```x_z_PT_GNN_NS_MCS.txt``` - SL-GNN-NS-MCS.
Here, ```z``` corresponds to the system under evaluation.

## Citation
If you use this code, dataset, or the proposed methodology in academic research, please cite the accompanying paper:
```
@article{Assis2026,
  author  = {Fernando A. Assis and Marcos Netto and Arnob Ghosh},
  title   = {Accelerated Composite Reliability Assessment Using a Reinforcement Learning-Driven Graph Neural Network Surrogate},
  journal = {IEEE Transactions on Power Systems},
  year    = {2026}
}
```
