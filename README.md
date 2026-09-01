# Accelerated composite reliability assessment using a reinforcement learning-driven graph neural network surrogate

**Repository accompanying:**

"Accelerated composite reliability assessment using a reinforcement learning-driven graph neural network surrogate" by Fernando A. Assis, Marcos Netto, and Arnob Ghosh

---

## Overview
This repository provides the Python implementation of the framework proposed in our paper, which accelerates power system reliability assessment by combining Non-Sequential Monte Carlo Simulation (NS-MCS) with a reinforcement learning-driven Graph Neural Network (RL-GNN) surrogate model to speed up AC-OPF evaluations. It is intended for reproducibility and extension by other researchers.


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
git clone https://github.com/FAssisRepository/ReliabilityAssessment_RL/Proposed_Framework_RL_GNN
cd Proposed_Framework_RL_GNN
```

## Repository structure

```bash
.
├── Proposed_Framework_RL_GNN/        # Main code directory
	├── main.py                       # Main execution script running the full training and evaluation pipeline
	├── InputData/                    # Directory with system datasets, load curve, and central configuration file
		├── 00_SIMULATION_SETTINGS.txt    # Central configuration file (System selection and NS-MCS, GNN, and training settings)
		├── IEEERTS_LOAD.load             # Full-year hourly load profile curve
		├── SIST_6BUS_GLOBAL_AC.dat       # RBTS 6-bus system data
		├── IEEERTS79_GLOBAL_AC.dat       # IEEE-RTS 24-bus system data
		├── SIST_200B_GLOBAL_AC.dat       # Illinois 200-bus (ACTIVSg200) system data
	├── GNNmodel/                     # Directory for saving trained GNN model checkpoints
	├── OutputData/                   # Directory for storing simulation results and log files
    ├── input_data_class.py           # Data parser loading system topologies, network parameters, and settings
	├── auxiliar_classes.py           # Helper utilities and data structures for simulation logging and tracking
	├── DRL_reliab_env_1episode.py    # Single-episode RL environment handling graph states, actions, rewards, and constraints
	├── element_classes.py            # Object models for power system elements (buses, generators, circuits, loads)
	├── flow_AC.py                    # AC Power Flow (AC-PF) solver (without optimization)
	├── OPF_AC.py                     # AC Optimal Power Flow (AC-OPF) solver
	└── reliab_assessment.py          # Primary reliability assessment engine, GNN training, and evaluation tests
│
├── test_systems_scenarios_description.pdf   # Detailed description of the test systems and scenario generation
```

## Configuration

Simulation and training parameters are primarily defined in:
```
Proposed_Framework_RL_GNN/InputData/00_SIMULATION_SETTINGS.txt
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

## Running the Framework

The main execution script is:
```
Proposed_Framework_RL_GNN/main.py
```
The framework performs the corresponding training and reliability assessment procedures according to the settings specified in:
```
Proposed_Framework_RL_GNN/InputData/00_SIMULATION_SETTINGS.txt
```
Simulation results, log files, and trained model checkpoints are stored in the corresponding output directories.
```
Proposed_Framework_RL_GNN/OutputData/
```

## Reproducibility
To reproduce the experiments reported in the accompanying paper:
- Install the required Python dependencies.
- Clone the repository Proposed_Framework_RL_GNN/.
- Select the desired test system and simulation parameters in InputData/00_SIMULATION_SETTINGS.txt.
- Run main.py.
- Use the generated output files in OutputData/ directory to check the results (training dynamics, corrective performance metrics, and reliability indices).

For a detailed description of the test systems and scenario-generation procedures, see:
```
test_systems_scenarios_description.pdf
```

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
