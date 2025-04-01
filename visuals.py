import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpi4py import MPI
from itertools import product




def nudging_reservoir_visuals(path, noise_variance, nudging_strength):

    # Load in data from folder
    vpt = np.load(f'{path}vpt.npy')
    best_pred_states = np.load(f'{path}best_pred_states.npy')
    best_U_pred = np.load(f'{path}best_U_pred.npy')
    U_actual = np.load(f'{path}U_actual.npy')
    t = np.linspace(0,10,1001)

    
    plt.figure(figsize=(14,6))
    plt.suptitle(f"Reservoir Nudging with Noise Variance={noise_variance}, Nudging Strength={nudging_strength}")
    
    # Plot the Internal Reservoir States
    plt.subplot(211)
    plt.title("Internal States")
    color = np.random.rand(best_pred_states.shape[1], 3)
    for i, c in enumerate(color):
        plt.plot(t, best_pred_states[:,i], c=c)
    plt.axvline(np.max(vpt), color='green', label='VPT')

    # Plot the Recompiled Attractor vs Actual
    plt.subplot(212)
    plt.title("Prediction")
    for i in range(3):
        if i == 0:
            plt.plot(t, best_U_pred[i,:], c="orange", label="Predicted")
        else:
            plt.plot(t, best_U_pred[i,:], c="orange")

    for i in range(3):
        if i == 0:
            plt.plot(t, U_actual[:,i], c="blue", label="Actual")
        else:
            plt.plot(t, U_actual[:,i], c="blue")

    plt.axvline(np.max(vpt), color='green', label='VPT')
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'{path}plot.png')



def nudging_reservoir_visuals_mpi_helper():

    noise_variances = [0,1e-7,1e-3]
    nudging_strengths = [0,1e-8, 1e-4, 1, 10]

    # Establish MPI Connections
    RANK = MPI.COMM_WORLD.Get_rank()
    run_combinations = list(product(noise_variances, nudging_strengths))
    noise_variance, nudging_strength = run_combinations[RANK]

    # Designate path for thread to run
    path = f'nudging_reservoirs_results/{noise_variance}_{nudging_strength}/'

    nudging_reservoir_visuals(path, noise_variance, nudging_strength)