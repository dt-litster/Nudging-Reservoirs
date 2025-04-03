
import torch.nn as nn
import chaosode
import torch
from torchdiffeq import odeint
import argparse
from tqdm import tqdm
from scipy.interpolate import CubicSpline
import numpy as np
from pathlib import Path

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



class NODE(nn.Module):
    def __init__(self, width, add_in_bias=True, add_out_bias=True, in_layers=3, out_layers=3):
        super(NODE, self).__init__()
        self.num_calls = 0
        self.quad = in_layers==12
        self.net = nn.Sequential(
            nn.Linear(in_layers, width, bias=add_in_bias),
            nn.Tanh(),
            nn.Linear(width, out_layers, bias=add_out_bias),
        )

    def forward(self, t, y):
        # y has shape (batch_size, 4)
        self.num_calls += 1
        if self.quad:
            return self.net(torch.cat([y, torch.outer(y, y).reshape(-1)]))
        return self.net(y)

# Get command line arguments for the filepath, model type, and various params
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help="filepath in which to save losses and weights. the script will automatically append the width of the neural net and the number of epochs trained")
parser.add_argument('-lr', '--lr', type=float, default=1e-3, help="learning rate for Adam optimizer")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay for Adam optimizer")
parser.add_argument('-ws', '--window_schedule', type=str, default="5", help="a schedule for window size used in training samples. should be a string of the form 't**2' that could be put in a lambda function of one variable t. the result will be clipped between 1 and 90 and be cast to an int to avoid errors. defaults to constant window of width 5.")
parser.add_argument('--num_epochs', type=int, default=1000, help="num of epochs to train for")
parser.add_argument('--save_every', type=int, default=100, help="how often to save the weights and losses")
parser.add_argument('--val_every', type=int, default=25, help="how often to test against the validation data")
parser.add_argument('-w', '--width', nargs='+', type=int)
parser.add_argument('-c', '--cuda', action="store_true", help="use cuda if available")
parser.add_argument('--bias', type=str, choices=["both", "first", "last", "none"], default="both")
parser.add_argument('-q', '--quad', action='store_true', help="feed quadratic terms in to node")
args = parser.parse_args()

# Create inidividual path for this run
path = args.path + f'{WINDOW_FNS[args.window_schedule]}_{LRS[args.lr]}_{WEIGHT_DECAYS[args.weight_decay]}/'
directory = Path(path)
directory.mkdir(parents=True, exist_ok=True)

# Set model, optimizer and loss
device = 'cpu'
if args.cuda:
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        print("cuda is not available; defaulting to cpu")

window_fn = lambda t: max(2, min(9000, int(eval(args.window_schedule))))

loss_fn = nn.MSELoss()
biases = (args.bias in {"both", "first"}, args.bias in {"both", "last"})
in_width = 3 if not args.quad else 12

for i, width in enumerate(args.width):
    model = NODE(width, *biases, in_layers=in_width).to(device)
    t, U = chaosode.orbit("lorenz", duration=100)
    u = CubicSpline(t, U)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    losses = []

    # Training
    train_losses = []
    val_losses = []

    X, y = torch.from_numpy(u(t[:9000])), torch.from_numpy(u(t[9000:10000]))
    t = torch.from_numpy(t).to(device)
    torch.save(X, f'{path}node_{width}_X.pt')
    torch.save(y, f'{path}node_{width}_y.pt')
    for idx in tqdm(range(args.num_epochs), initial=i*args.num_epochs, total=args.num_epochs*len(args.width), leave=True):
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

        if (idx + 1) % args.save_every == 0:
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
