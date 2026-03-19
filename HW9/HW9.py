import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from scipy.integrate import solve_ivp


## Data Setup ##
####################################################################
def data_setup():

    # ------------ import data ------------
    # 1000 trajectory sets with 100 time steps split 75/25 for training/testing
    data = np.load("spring_data.npz")
    # X_train (75000, 4, 5)  data points, 4 particles, 5 states: x, y, Vx, Vy, m
    X_train = torch.tensor(data['X_train'], dtype=torch.float32)
    # y_train (75000, 4, 2)  data points, 4 particles, 2 states: ax, ay
    y_train = torch.tensor(data['y_train'], dtype=torch.float32)
    # X_test (25000, 4, 5)  data points, 4 particles, 5 states
    X_test = torch.tensor(data['X_test'], dtype=torch.float32)
    # y_test (25000, 4, 2)  data points, 4 particles, 2 states
    y_test = torch.tensor(data['y_test'], dtype=torch.float32)
    # 100 time steps
    times = torch.tensor(data['times'], dtype=torch.float32)

    # ------- Save a few trajectories for plotting -------
    # the data points are currently ordered in time (for each separate trajectory)
    # so I'm going to save one set before shuffling the data.
    nt = len(times)
    train_traj = X_train[:nt, :, :]   # (100, 4, 5)
    test_traj  = X_test[:nt, :, :]    # (100, 4, 5)

    # You can comment this out, just showing you how to do this
    # for when you'll want to compare later.
    plt.figure()
    for j in range(4):
        plt.plot(train_traj[:, j, 0], train_traj[:, j, 1])
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.title('True Training Trajectory')

    plt.figure()
    for j in range(4):
        plt.plot(test_traj[:, j, 0], test_traj[:, j, 1])
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.title('True Testing Trajectory')

    plt.show()

    # ------ edge index ------
    # each of the 4 particles interacts with every other particle
    edge_index = torch.tensor([
        [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2]
    ], dtype=torch.long)

    # -------- Build PyG Data objects ----------
    # Each data point: x is (4, 5) node features, y is (4, 2) target accelerations
    # Use a subset for faster development; full dataset ~75k train / 25k test
    n_train = 30000   # feel free to tune
    n_test  = 5000

    train_list = []
    for i in range(n_train):
        d = Data(x=X_train[i], y=y_train[i], edge_index=edge_index)
        train_list.append(d)

    test_list = []
    for i in range(n_test):
        d = Data(x=X_test[i], y=y_test[i], edge_index=edge_index)
        test_list.append(d)

    train_loader = DataLoader(train_list, batch_size=1000, shuffle=True)
    test_loader  = DataLoader(test_list,  batch_size=1000, shuffle=False)

    return train_loader, test_loader, train_traj, test_traj, times, edge_index


## Model Architecture ##
####################################################################
# Graph Network following Cranmer et al. 2020
# Key design choices (per instructor tips):
#   - msg_dim = 2  (bottleneck instead of L1 sparsity)
#   - smaller hidden size since msg_dim is already small
#   - MAE loss (not MSE)
class GN(MessagePassing):
    def __init__(self, n_f=5, msg_dim=2, ndim=2, hidden=100, aggr='add'):
        super(GN, self).__init__(aggr=aggr)

        # Message function: takes features of sender + receiver → message
        # Input: 2*n_f (concatenated xi, xj), Output: msg_dim
        self.msg_fnc = Seq(
            Lin(2 * n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, msg_dim),
        )

        # Node update function: aggregated messages + node features → acceleration
        # Input: msg_dim + n_f, Output: ndim (ax, ay)
        self.node_fnc = Seq(
            Lin(msg_dim + n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, ndim),
        )

    def forward(self, x, edge_index):
        # x: (n_nodes_in_batch, n_f)
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j):
        # x_i, x_j: (n_edges, n_f)
        tmp = torch.cat([x_i, x_j], dim=1)   # (n_edges, 2*n_f)
        return self.msg_fnc(tmp)              # (n_edges, msg_dim)

    def update(self, aggr_out, x=None):
        # aggr_out: (n_nodes, msg_dim) — summed messages
        tmp = torch.cat([x, aggr_out], dim=1)   # (n_nodes, n_f + msg_dim)
        return self.node_fnc(tmp)               # (n_nodes, ndim)


## Loss Function ##
####################################################################
def loss_fn(pred, target):
    # MAE as in the paper
    return torch.mean(torch.abs(pred - target))


## Training Loop ##
####################################################################
def train(model, optimizer, train_loader, test_loader, epochs, device):

    train_losses = []
    test_losses  = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            pred = model(batch.x, batch.edge_index)   # (n_nodes_total, 2)
            loss = loss_fn(pred, batch.y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_train = epoch_loss / n_batches

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            nb = 0
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index)
                test_loss += loss_fn(pred, batch.y).item()
                nb += 1
            avg_test = test_loss / nb

        train_losses.append(avg_train)
        test_losses.append(avg_test)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | Train MAE: {avg_train:.4e} | Test MAE: {avg_test:.4e}")

    return train_losses, test_losses


## Trajectory Rollout via solve_ivp ##
####################################################################
def rollout_trajectory(model, traj_np, times_np, edge_index, device):
    """
    Use the trained GNN to predict accelerations, then integrate with solve_ivp.

    traj_np  : (nt, 4, 5)  true trajectory (x, y, vx, vy, m)
    times_np : (nt,)        time vector
    Returns predicted positions shape (nt_dense, 4, 2).
    """
    model.eval()

    # Extract mass (constant along trajectory) and initial state
    mass = traj_np[0, :, 4]   # (4,)

    # State vector for solve_ivp: [x0,y0,x1,y1,..., vx0,vy0,vx1,vy1,...]
    # i.e., 4*2 positions + 4*2 velocities = 16 values
    pos0 = traj_np[0, :, :2].flatten()   # (8,)
    vel0 = traj_np[0, :, 2:4].flatten()  # (8,)
    state0 = np.concatenate([pos0, vel0]) # (16,)

    def ode_rhs(t, state):
        # Unpack state
        pos = state[:8].reshape(4, 2)   # (4, 2)
        vel = state[8:].reshape(4, 2)   # (4, 2)

        # Build node feature tensor: (4, 5) = [x, y, vx, vy, m]
        node_feat = np.concatenate([pos, vel, mass[:, None]], axis=1)  # (4, 5)
        x = torch.tensor(node_feat, dtype=torch.float32).to(device)

        with torch.no_grad():
            acc = model(x, edge_index.to(device)).cpu().numpy()  # (4, 2)

        dpos_dt = vel.flatten()    # velocities are derivative of positions
        dvel_dt = acc.flatten()    # accelerations are derivative of velocities
        return np.concatenate([dpos_dt, dvel_dt])

    t_span = (times_np[0], times_np[-1])
    sol = solve_ivp(ode_rhs, t_span, state0, t_eval=times_np, method='RK45',
                    rtol=1e-6, atol=1e-8)

    # sol.y is (16, nt); reshape to positions (nt, 4, 2)
    pos_pred = sol.y[:8, :].T.reshape(-1, 4, 2)   # (nt, 4, 2)
    return pos_pred


## Main ##
####################################################################
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader, test_loader, train_traj, test_traj, times, edge_index = data_setup()

    # Initialize model and optimizer
    model = GN(n_f=5, msg_dim=2, ndim=2, hidden=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100
    train_losses, test_losses = train(model, optimizer, train_loader, test_loader, epochs, device)

    # ---- Loss plot ----
    plt.figure()
    plt.plot(range(epochs), train_losses, label='Train MAE')
    plt.plot(range(epochs), test_losses,  label='Test MAE', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_history.png', dpi=150)
    plt.show()

    # Trajectory rollout and comparison
    times_np = times.numpy()

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # Training trajectory comparison 
    train_traj_np = train_traj.numpy()   # (100, 4, 5)
    pred_train = rollout_trajectory(model, train_traj_np, times_np, edge_index, device)

    plt.figure(figsize=(6, 6))
    for j in range(4):
        plt.plot(train_traj_np[:, j, 0], train_traj_np[:, j, 1],
                 color=colors[j], linewidth=2, label=f'True P{j}')
        plt.plot(pred_train[:, j, 0], pred_train[:, j, 1],
                 color=colors[j], linewidth=1.5, linestyle='--', label=f'Pred P{j}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Training Trajectory: True vs Predicted')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig('train_traj_comparison.png', dpi=150)
    plt.show()

    # Testing trajectory comparison
    test_traj_np = test_traj.numpy()    # (100, 4, 5)
    pred_test = rollout_trajectory(model, test_traj_np, times_np, edge_index, device)

    plt.figure(figsize=(6, 6))
    for j in range(4):
        plt.plot(test_traj_np[:, j, 0], test_traj_np[:, j, 1],
                 color=colors[j], linewidth=2, label=f'True P{j}')
        plt.plot(pred_test[:, j, 0], pred_test[:, j, 1],
                 color=colors[j], linewidth=1.5, linestyle='--', label=f'Pred P{j}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Testing Trajectory: True vs Predicted')
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('test_traj_comparison.png', dpi=150)
    plt.show()