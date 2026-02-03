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
        
        # Define learned parameters as members of the class
        # Lambda 2 needs to be nagtive so the optimizer does not break the physics
        self.lambda1 = nn.Parameter(torch.tensor(0.0))
        self.lambda2 = nn.Parameter(torch.tensor(-5.0))

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

    # Ensure lambda2 is negative
    lambda2 = -torch.exp(model.lambda2)

    # Compute residual
    return dudt + model.lambda1 * u*dudx + lambda2 * d2udx2

def inputs(n_coll, device):

    # Input data
    data = np.loadtxt("burgers.txt")

    # Move data to tansors that can be use on the GPU
    x_data = torch.tensor(data[:, 0:1], dtype=torch.float32, device=device)
    t_data = torch.tensor(data[:, 1:2], dtype=torch.float32, device=device)
    u_data = torch.tensor(data[:, 2:3], dtype=torch.float32, device=device)

    # Define collocation points and use Latin Hypercube samplig
    lower_bounds = [0, -1]
    upper_bounds = [1, 1]
    sampler = qmc.LatinHypercube(d=2)

    coll_np = qmc.scale(sampler.random(n_coll), lower_bounds, upper_bounds)  # (n_coll, 2) numpy
    coll = torch.tensor(coll_np, dtype=torch.float32, device=device)         # move to GPU

    t_func = coll[:, 0:1]
    x_func = coll[:, 1:2]

    return x_data, t_data, u_data, t_func, x_func

def loss(func_residuals, pred, data):

    # Calculate MSE for boundary conditions and physics residuals
    MSEu = torch.mean((pred - data)**2)
    MSEf = torch.mean((func_residuals)**2)
    Loss = MSEu + 0.01*MSEf

    return Loss

def train(model, optimizer, epochs, n_coll, device):

    # Put in training mode
    model.train()

    # Get the inputs for training
    x_data, t_data, u_data, t_func, x_func = inputs(n_coll, device)
    
    # Crate a list for storing training loss
    train_loss = []
    

    for epoch in range(epochs):

        optimizer.zero_grad()

        # Get the collocation points
        _, _, _, t_func, x_func = inputs(n_coll, device) 
        # Evaluate the model
        u_eval = model(t_data, x_data)

        # Calculate the residuals for the physics loss
        f = residual(model, (t_func, x_func))

        # Calculate the total loss
        loss_val = loss(f, u_eval, u_data)

        # Backpropagation
        loss_val.backward()

        # Take an optimization step
        optimizer.step()

        # Add the new training loss to the list for plotting
        train_loss.append(loss_val.item())

        # Print the progress every 100 epochs
        if epoch % 100 == 0:
            lam1 = model.lambda1.item()
            lam2 = (-torch.exp(model.lambda2)).item()
            print(f"epoch={epoch} loss={loss_val.item():.3e} lambda1={lam1:.4f} lambda2={lam2:.6f}")

    return train_loss

# I had ChatGPT write this so it looked good
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
    plt.savefig('pinn_solution.png')
    plt.show()

if __name__ == "__main__":

    # Initialize model, optimizer, and training parameters
    epochs = 2000
    n_coll = 10000

    # Check for a GPU or CUDA device (way faster than CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create a model instance
    model = PINN(hlayers = 8, width = 40).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)
    
    # Train the model
    x = train(model, optimizer, epochs, n_coll, device)

    # Plot the results
    t = range(1, epochs + 1)
    plot_pinn_map(model, device=device)

    # Print the learned parameters
    print(f'Lambda1: {model.lambda1.item()}')
    print(f'Lambda2: {(-torch.exp(model.lambda2)).item()}')

    # Print distance from the provided parameter values
    print(f'Distance from true lambda1: {abs(model.lambda1.item() - 1.0)}')
    print(f'Distance from true lambda2: {abs((-torch.exp(model.lambda2)).item() - (-0.01/np.pi))}')

    # Plot training loss
    plt.plot(t, x)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.savefig('training_loss.png')
    plt.show()