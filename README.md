# Nudging-Reservoirs
This repository contains preliminary code and ideas for implementing Reservoir Computers with Nudging techniques and comparing the results to Neural ODEs restricted to a similar setup. Our results indicate that while reservoirs may look superficially like approximations to nudged NODEs, this connection does not seem genuine since for a fixed size, reservoirs tend to outperform nudged NODEs. 

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)

## Features
There are three different components to this repository.
1. Helper/Misc. Files
    - chaosode.py: Simulate chaotic dynamical systems.
    - utils.py: Calculate the valid prediction time of the forecasted systems.
    - visuals.py: Plot the numerical results of the various experiments.
    - scripts/: Contains scripts for running numerical simulations on supercomputers.
    - Lecture_Notes.tex: LaTeX file for the lecture notes to be given in class.
    - Reservoir Nudging.tex: LaTeX file with initial musings and theory exploration.
2. Nudged Reservoir Computing
    - Nudging_example.ipynb: Base nudging of the chatoic dynamical system (no Reservoir Computing).
    - nudged_reservoir_computing.ipynb: Worked out example with visuals demonstrating nudging involved in the forecast stage of Reservoir Computing.
    - nudged_reservoir.py: Generalized Reservoir Nudging code designed to be ran on the supercomputer for numerical simulations.
    - nudging_reservoir_results/: Output directory containing numerical experiments of different nudged reservoirs along with plots of the forecast and valid prediction time.
3. Neural ODE
    - train_node.py: Pytorch implementation of a Neural ODE to approximate the Lorenz system using a similar amount of nodes as the Reservoir Computer simulations.
    - train_node_parallelized.py: Parallelized train_node.py to handle multiple simulations per job run.
    - node_results/: Output directory for the Pytorch simulations and results.

## Installation
Steps for downloading the Github Repo and setting up a working conda environment.

```bash
git clone https://github.com/dt-litster/Nudging-Reservoirs.git
cd Nudging-Reservoirs
conda create -n torch_reservoir
conda activate torch_reservoir
conda install openmpi mpi4py jupyter numpy scipy matplotlib pytorch torchdiffeq tqdm
```

The scripts will also need to be updated to your current system (change pathing, email, processor #, etc.)
* scripts/nudged_reservoir_job_script.sh
* scripts/nudged_reservoir_visuals.sh
* scripts/train_node.sh
* scripts/train_node_job_script.sh
* scripts/train_node_job_script_parallelized.sh

## Usage
Once this is done, you should be good to run it on the supercomputer.  We have slurm scripts setup in the scripts/ directory to use.
To run the test script, simply type the following in the terminal:

```bash
sbatch scripts/nudged_reservoir_job_script.sh
```

This should generate a decent amount of initial test data to run the visualization script on to see what you're working with:

```bash
sbatch scripts/nudged_reservoir_visuals.sh
```

Preliminary results will be stored in a *results/ folder in the directory.

## Credits

For more information see:
* https://acme.byu.edu/00000180-6d94-d2d1-ade4-6ff4c7cf0001/mpi (for a decent walkthrough of mpi basic principles)
* https://rc.byu.edu/wiki/?id=Slurm (for more information on Slurm scripts - the site in general is good if you're operating on the BYU supercomputer)
* https://github.com/djpasseyjr/rescomp (Reservoir Code and Chaosode.py examples)
