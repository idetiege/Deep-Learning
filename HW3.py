import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.stats import qmc

class PINN(nn.Module):
    def __init__(self, hlayers, width):
        super(PINN, self).__init__()

        layers = []
        layers.append(nn.Linear(2, width))
        layers.append(nn.Tanh())

        for _ in range(hlayers-1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, 1))

        self.network = nn.Sequential(*layers)
        
    def forward(self, t, x): # t is size 1,2
        return self.network(torch.cat([t, x], dim=1))
    
def residual(model, params): # t size: (ncol, 2)

    # Unpack parameters
    t, x = params

    # We need the gradients
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)

    # Evaluate model
    u = model(t, x) # y size: (ncol, 2)
    
    # Compute derivatives
    dudt = torch.autograd.grad(u, t, grad_outputs= torch.ones_like(u), create_graph=True)[0]
    dudx = torch.autograd.grad(u, x, grad_outputs= torch.ones_like(u), create_graph=True)[0]
    d2udx2 = torch.autograd.grad(dudx, x, grad_outputs= torch.ones_like(dudx), create_graph= True)[0]

    # Compute residual
    return dudt + u*dudx - (0.01/torch.pi) * d2udx2

def inputs(n_coll):

    # Define boundary conditions
    # The size of out tensors are T1: (50,2), T2: (25,2), T3: (25,2)
    # U1: (50,1), U2: (25,1), U3: (25,1)
    t0 = torch.zeros((50, 1))
    x0 = -1 + 2*torch.rand((50, 1))
    U1 = -torch.sin(torch.pi * x0)
    T1 = torch.cat((t0, x0), dim = 1)

    t1 = torch.rand((25, 1))
    x1 = -torch.ones((25, 1))
    U2 = torch.zeros((25, 1))
    T2 = torch.cat((t1, x1), dim = 1)

    t2 = torch.rand((25, 1))
    x2 = torch.ones((25, 1))
    U3 = torch.zeros((25, 1))
    T3 = torch.cat((t2, x2), dim = 1)

    # Create the two large tensors for boundary conditions
    # Xbc is a tensor containing all the (t,x) pairs for the boundary conditions
    # Ubc is a tensor containing all the u values for the boundary conditions
    Xbc = torch.cat((T1, T2, T3), dim = 0)
    Ubc = torch.cat((U1, U2, U3), dim = 0)

    lower_bounds = [0, -1]
    upper_bounds = [1, 1]
    sampler = qmc.LatinHypercube(d=2)

    coll = qmc.scale(sampler.random(n_coll), lower_bounds, upper_bounds)
    coll = torch.tensor(coll, dtype = torch.float32)

    # Split collocation points into t and x
    t_func = coll[:, 0:1]
    x_func = coll[:, 1:2]

    return Xbc, Ubc, t_func, x_func

def loss(func_residuals, boundary_residuals, ubc):

    # Calculate MSE for boundary conditions and physics residuals
    MSEu = torch.mean((boundary_residuals - ubc)**2)
    MSEf = torch.mean((func_residuals)**2)
    Loss = MSEu + MSEf

    return Loss

def train(model, optimizer, epochs, n_coll):

    # Put in training mode
    model.train()

    # Get the inputs for training
    Xbc, Ubc, t_func, x_func = inputs(n_coll)
    
    # Crate a list for storing training loss
    train_loss = []
    
    t_bc = Xbc[:, 0:1]
    x_bc = Xbc[:, 1:2]

    for _ in range(epochs):

        optimizer.zero_grad()

        # Evaluate the model
        ubc = model(t_bc, x_bc)

        # Calculate the residuals for the physics loss
        f = residual(model, (t_func, x_func))

        # Calculate the total loss
        loss_val = loss(f, ubc, Ubc)

        # Backpropagation
        loss_val.backward()

        # Omtimizer steps and updates the parameters
        optimizer.step()

        # Add the new training loss to the list for plotting
        train_loss.append(loss_val.item())

    return train_loss

def plot_pinn_map(model, Xbc=None, u_exact_fn=None,
                  t_min=0.0, t_max=1.0, x_min=-1.0, x_max=1.0,
                  Nt=200, Nx=200, slice_ts=(0.25, 0.50, 0.75),
                  device=None):
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # ----- build grid -----
    t = np.linspace(t_min, t_max, Nt)
    x = np.linspace(x_min, x_max, Nx)
    T, X = np.meshgrid(t, x)                # shapes (Nx, Nt)

    t_flat = torch.tensor(T.reshape(-1, 1), dtype=torch.float32, device=device)
    x_flat = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=device)

    # ----- model prediction on grid -----
    with torch.no_grad():
        u_pred = model(t_flat, x_flat).cpu().numpy().reshape(Nx, Nt)

    # ----- exact (optional) -----
    u_exact = None
    if u_exact_fn is not None:
        u_exact = u_exact_fn(T, X)          # must return shape (Nx, Nt)

    # ----- figure layout -----
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.3], hspace=0.45, wspace=0.35)

    # ===== top heatmap =====
    ax0 = fig.add_subplot(gs[0, :])
    im = ax0.pcolormesh(T, X, u_pred, shading="auto", cmap="rainbow", vmin =-1, vmax = 1)
    cbar = fig.colorbar(im, ax=ax0, pad=0.02)
    ax0.set_title(r"$u(t,x)$")
    ax0.set_xlabel(r"$t$")
    ax0.set_ylabel(r"$x$")

    # overlay training/data points if provided
    if Xbc is not None:
        Xbc_np = Xbc.detach().cpu().numpy() if torch.is_tensor(Xbc) else np.asarray(Xbc)
        t_data = Xbc_np[:, 0]
        x_data = Xbc_np[:, 1]
        ax0.scatter(t_data, x_data, s=25, marker="x", linewidths=1.5, label=f"Data ({len(t_data)} points)")
        ax0.legend(loc="upper right")

    # ===== bottom slices =====
    for j, ts in enumerate(slice_ts):
        ax = fig.add_subplot(gs[1, j])

        # evaluate model on slice line x∈[x_min,x_max] at fixed t=ts
        x_line = np.linspace(x_min, x_max, 400).reshape(-1, 1)
        t_line = np.full_like(x_line, ts)

        t_t = torch.tensor(t_line, dtype=torch.float32, device=device)
        x_t = torch.tensor(x_line, dtype=torch.float32, device=device)

        with torch.no_grad():
            u_line = model(t_t, x_t).cpu().numpy().flatten()

        ax.plot(x_line.flatten(), u_line, label="Prediction", linestyle="--")

        if u_exact_fn is not None:
            uex_line = u_exact_fn(t_line, x_line).flatten()
            ax.plot(x_line.flatten(), uex_line, label="Exact")

        ax.set_title(fr"$t = {ts:.2f}$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u(t,x)$")
        ax.grid(True, alpha=0.3)
        if j == 1:
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.55), ncol=2, frameon=False)

    plt.show()

if __name__ == "__main__":

    # Initialize model, optimizer, and training parameters
    epochs = 1000
    n_coll = 10000
    model = PINN(hlayers = 8, width = 30)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    # Train the model
    x = train(model, optimizer, epochs, n_coll)

    # Plot the results
    t = range(1, epochs + 1)
    plot_pinn_map(model)

    plt.plot(t, x)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')

    plt.show()