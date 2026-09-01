# Accelerated composite reliability assessment using a reinforcement learning-driven graph neural network surrogate

**Repository accompanying:**

"Accelerated composite reliability assessment using a reinforcement learning-driven graph neural network surrogate" by Fernando A. Assis, Marcos Netto, and Arnob Ghosh

---

## Overview
This repository provides the Python implementation of the framework proposed in our paper, which accelerates power system reliability assessment by combining Non-Sequential Monte Carlo Simulation (NS-MCS) with a reinforcement learning-driven Graph Neural Network (RL-GNN) surrogate model to speed up AC-OPF evaluations. It is intended for reproducibility and extension by other researchers.


## Prerequisites & Installation

1. **Requirements:**

Python 3.8+

PyTorch & PyTorch Geometric

Gymnasium

PYPOWER

SciPy

NumPy

2. **Quick Install:**

```pip install torch torch-geometric numpy scipy pypower gymnasium```

## Repository structure

```bash
.
├── 00_SIMULATION_SETTINGS    # Central configuration file (System selection and NS-MCS, GNN, and training settings)
├── main.py                   # Main execution script running the full training and evaluation pipeline
    ├── input_data_class.py           # Data parser loading system topologies, network parameters, and settings
	├── auxiliar_classes.py           # Helper utilities and data structures for simulation logging and tracking
	├── DRL_reliab_env_1episode.py    # Single-episode RL environment handling graph states, actions, rewards, and constraints
	├── element_classes.py            # Object models for power system elements (buses, generators, circuits, loads)
	├── flow_AC.py                    # AC Power Flow (AC-PF) solver (without optimization)
	├── OPF_AC.py                     # AC Optimal Power Flow (AC-OPF) solver
	└── reliab_assessment.py          # Primary reliability assessment engine, GNN training, and evaluation tests
```
