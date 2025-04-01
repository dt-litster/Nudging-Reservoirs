import chaosode
from scipy.interpolate import CubicSpline
from scipy import integrate
import numpy as np
from matplotlib import pyplot as plt
import utils  
from pathlib import Path
from mpi4py import MPI
from itertools import product
from pathlib import Path
import time

RANK = MPI.COMM_WORLD.Get_rank()

# Code to handle parallelizing
noise_variances = [0,1e-7,1e-3]
nudging_strengths = [0,1e-8, 1e-4, 1, 10]
run_combinations = list(product(noise_variances, nudging_strengths))
noise_variance, nudging_strength = run_combinations[RANK]
n = 50  # Number of reservoir nodes
gamma = 5  # Reservoir relaxation rate
sigma = 0.14  # Input scaling
mean_degree = 1
density = mean_degree / n
alpha = 1e-6  # Regularization parameter for training
run_total = 30
vpts = np.zeros(run_total)
best_vpt = 0 
best_pred_states = None
best_U_pred = None

# Time for early ending if needed
tf = 9500
t0 = time.time()

# Ensure Path Exists
path = f'nudging_reservoirs_results/{noise_variance}_{nudging_strength}/'
directory = Path(path)
directory.mkdir(parents=True, exist_ok=True)

# Load Lorenz system data
t, U = chaosode.orbit("lorenz", duration=100)

# Interpolate data
u = CubicSpline(t, U) 
U_train = u(t[:9000])  # Observational data for training

#Create the true observation with multivariate normal random noise
def u0(t):
    return u(t) + np.random.multivariate_normal([0,0,0], noise_variance*np.eye(3), 1)

for i in range(run_total):
    
    # Check time and break if out of time
    t1 = time.time()
    if t1 - t0 > tf:
        print(f"Break in Combo: {noise_variance}_{nudging_strength}, iteration: {i}")
        break

    # Directed Erdos-Renyi adjacency matrix
    A = (np.random.rand(n, n) < density).astype(float)
    # Fixed random matrix for input coupling
    W_in = np.random.rand(n, 3) - 0.5


    def drdt(r, t):
        return gamma * (-r + np.tanh(A @ r + sigma * W_in @ u(t)))

    # Initial reservoir state
    r0 = np.random.rand(n)

    # Solve the reservoir dynamics during the training phase
    states = integrate.odeint(drdt, r0, t[:9000])

    # Training step: project training data onto reservoir states
    W_out = U_train.T @ states @ np.linalg.inv(states.T @ states + alpha * np.eye(states.shape[1]))

    # Prediction ODE IVP definition nudging during the prediction phase
    def trained_drdt(r, t):
        r = r.reshape((-1,1))
        return (gamma * (-r + np.tanh(A @ r + sigma * W_in @ (W_out @ r + nudging_strength * (u0(t).T - W_out @ r))))).flatten()

    # Initial state for prediction
    r0_pred = states[-1, :]  # Use the final training state


    # Solve for prediction phase
    pred_states = integrate.odeint(trained_drdt, r0_pred, t[9000:])

    # Map reservoir states onto the dynamical system space
    U_pred = W_out @ pred_states.T

    # Calculate Valid Prediction Time (VPT)
    test_t = t[9000:]
    vpt = utils.vpt_time(test_t, U_pred.T, u(test_t), vpt_tol=5)
    vpts[i] = (vpt)

    # Update best so far
    if vpt > best_vpt:
        best_vpt = vpt
        best_pred_states = pred_states
        best_U_pred = U_pred

# Save the data
np.save(f'{path}vpt.npy', vpts)
np.save(f'{path}best_pred_states.npy', best_pred_states)
np.save(f'{path}best_U_pred.npy', best_U_pred)
np.save(f'{path}U_actual.npy', U[9000:])