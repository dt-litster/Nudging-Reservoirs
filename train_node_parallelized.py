
import torch.nn as nn
import chaosode
import torch
from torchdiffeq import odeint
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import numpy as np
from pathlib import Path
from mpi4py import MPI
from itertools import product

# Dictionaries for file path naming

WINDOW_FNS = {
    "5": 0, 
    "10": 1, 
    "20": 2, 
    "40": 3,
    "min(10, t/500)": 4, 
    "min(20, t/500)": 5, 
    "min(40, t/500)": 6,
    "max(5,5-5*np.cos(t*np.pi/2500))": 7, 
    "max(5,10-10*np.cos(t*np.pi/2500))": 8, 
    "max(5,20-20*np.cos(t*np.pi/2500))": 9
}

LRS = {
    1e-3: 0,
    1e-4: 1,
    1e-5: 2
}

WEIGHT_DECAYS = {
    1e-1: 0,
    1e-3: 1,
    1e-5: 2
}

WIDTHS = {
    10: 0,
    50: 1,
    100: 2,
    5000: 3
}



class NODE(nn.Module):
    def __init__(self, width, add_in_bias=True, add_out_bias=True, in_layers=3, out_layers=3):
        super(NODE, self).__init__()
        self.num_calls = 0
        self.net = nn.Sequential(
            nn.Linear(in_layers, width, bias=add_in_bias),
            nn.Tanh(),
            nn.Linear(width, out_layers, bias=add_out_bias),
        )

    def forward(self, t, y):
        # y has shape (batch_size, 4)
        self.num_calls += 1
        return self.net(y)


# Code to handle parallelizing
SIZE = MPI.COMM_WORLD.Get_size()
RANK = MPI.COMM_WORLD.Get_rank()
run_combinations = list(product(WINDOW_FNS.keys(), LRS.keys(), WEIGHT_DECAYS.keys()))
window_schedule, lr, weight_decay= run_combinations[RANK]
bias = "both"
num_epochs = 30000
save_every = 6000

# Create inidividual path for this run
path = "node_results/" + f'{WINDOW_FNS[window_schedule]}_{LRS[lr]}_{WEIGHT_DECAYS[weight_decay]}/'
directory = Path(path)
directory.mkdir(parents=True, exist_ok=True)

# Set model, optimizer and loss
device = 'cpu'

window_fn = lambda t: max(2, min(9000, int(eval(window_schedule))))

loss_fn = nn.MSELoss()
biases = (bias in {"both", "first"}, bias in {"both", "last"})

for i, width in enumerate(WIDTHS.keys()):
    model = NODE(width, *biases).to(device)
    t, U = chaosode.orbit("lorenz", duration=100)
    u = CubicSpline(t, U)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []

    # Training
    train_losses = []
    val_losses = []

    X, y = torch.from_numpy(u(t[:9000])), torch.from_numpy(u(t[9000:10000]))
    t = torch.from_numpy(t).to(device)
    torch.save(X, f'{path}node_{width}_X.pt')
    torch.save(y, f'{path}node_{width}_y.pt')
    for idx in tqdm(range(num_epochs), initial=i*num_epochs, total=num_epochs, leave=True):
        optimizer.zero_grad()

        # get prediction and make a tuple to pass to the loss
        window = window_fn(idx)
        sample_t = np.random.randint(9000+1-window)
        X_batch = X[sample_t:sample_t+window].to(torch.float).to(device)
        X_pred = odeint(model, X_batch[0], t[sample_t:sample_t+window])

        # calculate loss and backprop
        loss = loss_fn(X_pred.squeeze(), X_batch.squeeze())  # MSELoss does a mean for us to make loss comparable across different window sizes
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if (idx + 1) % save_every == 0:
            model.eval()

            # test validation error
            optimizer.zero_grad()
            X_pred = odeint(model, X[0].to(torch.float), t[:9000])
            y_pred = odeint(model, X_pred[-1], t[9000:10000])
            loss = loss_fn(y_pred.squeeze(), y.squeeze())
            val_losses.append(loss.item())

            model.train()

            torch.save(model.state_dict(), f'{path}node_{width}_{idx+1}_weights.pt')
            torch.save(train_losses, f'{path}node_{width}_{idx+1}_train_losses.pt')
            torch.save(val_losses, f'{path}node_{width}_{idx+1}_val_losses.pt')
            torch.save(X_pred, f'{path}node_{width}_{idx+1}_X_pred.pt')
            torch.save(y_pred, f'{path}node_{width}_{idx+1}_y_pred.pt')
