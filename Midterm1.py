# Imports
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import qmc


# set random seed for reproducibility
seed = 42
torch.manual_seed(seed)

# Model definition
class PINN(nn.Module):
    def __init__(self, hlayers=2, width=20):
        super(PINN, self).__init__()

        layers = []
        layers.append(nn.Linear(1, width))
        # Try Tanh first and see if its good enough
        layers.append(nn.Tanh())

        for _ in range(hlayers-1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(width, 1))

        self.network = nn.Sequential(*layers)
        
    def forward(self, x): # x is size (n, 1)
        return self.network(x)

# Load data (use BCs and collocation points)
def data(n_coll):

    # Boundary condition points
    x_bc1 = torch.tensor([[0.0]], requires_grad=True) # 1x1
    x_bc2 = torch.tensor([[5.0]], requires_grad=True)   # 1x1
    y_bc1 = torch.tensor([[1.0]], requires_grad=True) # Size 1x1
    y_bc2 = torch.tensor([[0.0]], requires_grad=True) # Size 1x1

    # Colocation points using Latin Hypercube sampling
    lower_bounds = [0]
    upper_bounds = [5]
    sampler = qmc.LatinHypercube(d=1)
    coll_np = qmc.scale(sampler.random(n_coll), lower_bounds, upper_bounds)
    phys_coll = torch.tensor(coll_np, dtype=torch.float32) # Size n_coll x 1

    return x_bc1, x_bc2, y_bc1, y_bc2, phys_coll


# Loss function and residuals
def residual(model, x):
    x = x.clone().requires_grad_(True)
    
    y = model(x)
    dydx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    d2ydx2 = torch.autograd.grad(dydx, x, grad_outputs=torch.ones_like(dydx), create_graph=True)[0]

    return d2ydx2 - 2 * dydx + y - torch.cos(x)


def calc_loss(func_residuals, u_bc1, u_bc2, y_bc1, y_bc2):

    MSEbc1 = torch.mean((u_bc1 - y_bc1)**2)
    MSEbc2 = torch.mean((u_bc2 - y_bc2)**2)
    MSEf = torch.mean(func_residuals**2)

    # print(f"BC1: {MSEbc1.item()}, BC2: {MSEbc2.item()}, Physics Loss: {MSEf.item()}")
    return MSEbc1 + .1 * MSEbc2 + MSEf

# Training loop
def train(model, epochs, n_coll, optimizer ):
    model.train()
    x_bc1, x_bc2, y_bc1, y_bc2, _ = data(n_coll)
    
    training_loss = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Get collocation points
        _, _, _, _, phys_coll = data(n_coll)
        
        # Evalate model at boundary conditions
        u_bc1 = model(x_bc1)
        u_bc2 = model(x_bc2)
        # Calculate residuals for physics loss
        f = residual(model, phys_coll)

        # Calculate total loss
        loss_val = calc_loss(f, u_bc1, u_bc2, y_bc1, y_bc2)

        # Backprop
        loss_val.backward()
        optimizer.step()

        training_loss.append(loss_val.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss_val.item()}")

    return training_loss

if __name__ == "__main__":
    
    epochs = 10000
    n_coll = 1000
    # Create model instance
    model = PINN(hlayers=2, width=20)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Traing the model
    train_loss = train(model, epochs, n_coll, optimizer)

    # Plot training lsos over time for hyperparameter tuning and such
    # plt.figure(figsize=(10, 4))
    # plt.plot(train_loss)
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Training Loss")
    # plt.yscale("log")
    # plt.show()

    # Plot solution
    x_plot = torch.linspace(0, 5, 200).unsqueeze(1) # size needs to be n, 1 
    with torch.no_grad():
        y_plot = model(x_plot).squeeze() # size needs to be n for plotting
    
    # Plot
    plt.plot(x_plot.numpy(), y_plot.numpy(), label='PINN Solution')
    plt.scatter([0, 5], [1, 0], color='red', label='Boundary Conditions')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('PINN Solution')
    plt.legend()
    plt.savefig('pinn_solution.png')
    plt.show()


