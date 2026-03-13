import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW


## Load Data ##
######################################################################
# load low resolution data, which serves as input to our model
lfdata = np.load("sr_lfdata.npy")
lfx = lfdata[0, :, :]  # size 14 x 9  (height x width)
lfy = lfdata[1, :, :]
lfu = lfdata[4, :, :]
lfv = lfdata[5, :, :]

# plot the low resolution data (like fig 3a except we are using MRI noise here rather than Gaussian noise so it will look a bit different)
plt.figure()
plt.pcolormesh(lfx, lfy, np.sqrt(lfu**2 + lfv**2), cmap=cm.coolwarm, vmin=0.0, vmax=1.0)
plt.colorbar()

# load high resolution grids and mapping from low resolution to high resolution grid
hfdata = np.load("sr_hfdata.npy")
Jinv = hfdata[0, :, :]  # size 77 x 49 (height x width)
dxdxi = hfdata[1, :, :]
dxdeta = hfdata[2, :, :]
dydxi = hfdata[3, :, :]
dydeta = hfdata[4, :, :]
hfx = hfdata[5, :, :]
hfy = hfdata[6, :, :]

ny, nx = hfx.shape  #(77 x 49)
h = 0.01  # grid spacing in high fidelity (needed for derivatives)

# plt.show()

# see https://en.wikipedia.org/wiki/Finite_difference_coefficient
# or https://web.media.mit.edu/~crtaylor/calculator.html

# f should be a tensor of size: nbatch x nchannels x height (y or eta) x width (x or xi)
# This is written in a general way if one had more data, but for this case there is only 1 data sample,
# and there are only a few channels it might be clearer to you to separate the channels out into separate variables, 
# in which case the below could be simplified 
# (i.e., you remove the first two dimensions from everything so that input is just height x width if you desire).
def ddxi(f, h):
    # 5-pt stencil
    dfdx_central = (f[:, :, :, 0:-4] - 8*f[:, :, :, 1:-3] + 8*f[:, :, :, 3:-1] - f[:, :, :, 4:]) / (12*h)
    # 1-sided 4pt stencil
    dfdx_left = (-11*f[:, :, :, 0:2] + 18*f[:, :, :, 1:3] -9*f[:, :, :, 2:4] + 2*f[:, :, :, 3:5]) / (6*h)
    dfdx_right = (-2*f[:, :, :, -5:-3] + 9*f[:, :, :, -4:-2] -18*f[:, :, :, -3:-1] + 11*f[:, :, :, -2:]) / (6*h)

    return torch.cat((dfdx_left, dfdx_central, dfdx_right), dim=3)

def ddeta(f, h):
    # 5-pt stencil
    dfdy_central = (f[:, :, 0:-4, :] - 8*f[:, :, 1:-3, :] + 8*f[:, :, 3:-1, :] - f[:, :, 4:, :]) / (12*h)
    # 1-sided 4pt stencil
    dfdy_bot = (-11*f[:, :, 0:2, :] + 18*f[:, :, 1:3, :] -9*f[:, :, 2:4, :] + 2*f[:, :, 3:5, :]) / (6*h)
    dfdy_top = (-2*f[:, :, -5:-3, :] + 9*f[:, :, -4:-2, :] -18*f[:, :, -3:-1, :] + 11*f[:, :, -2:, :]) / (6*h)

    return torch.cat((dfdy_bot, dfdy_central, dfdy_top), dim=2)

## Data Prep ##
####################################################################

# Define tensors for Jinv, dxdxi, dxdeta, dydxi, dydeta, hfx, hfy, lfx, lfy, lfu, lfv
def prepare_data(lfdata, hfdata, device):
    lr_input = torch.from_numpy(lfdata).float().to(device)  # shape (1, 6, 14, 9)

    lr_input = lr_input.unsqueeze(0)  # add batch dimension, shape (1, 6, 14, 9)

    Jinv = torch.from_numpy(hfdata[0, :, :]).unsqueeze(0).unsqueeze(0).float().to(device)  # shape (77, 49)
    dxdxi = torch.from_numpy(hfdata[1, :, :]).unsqueeze(0).unsqueeze(0).float().to(device)
    dxdeta = torch.from_numpy(hfdata[2, :, :]).unsqueeze(0).unsqueeze(0).float().to(device)
    dydxi = torch.from_numpy(hfdata[3, :, :]).unsqueeze(0).unsqueeze(0).float().to(device)
    dydeta = torch.from_numpy(hfdata[4, :, :]).unsqueeze(0).unsqueeze(0).float().to(device)


    return lr_input, Jinv, dxdxi, dxdeta, dydxi, dydeta


## Model Architecture ##
####################################################################
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.upsample = nn.Upsample(size=(77, 49), mode='bicubic', align_corners=True)

        self.net = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=5, padding=2),
        )

    def forward(self, x):
        x = self.upsample(x)
        return self.net(x)

## Boundary Conditions ##
####################################################################
# Overwrite the network's predictions at the edges to satisfy the physics of the problem.
def apply_boundary_conditions(pred):
    # Pred shape: (1, 3, 77, 49) (nbatch x nchannels x height x width)
    # Channels are (u, v, p)
    u = pred[:, 0, :, :]
    v = pred[:, 1, :, :]
    p = pred[:, 2, :, :]

    # Bottom (inflow) boundary condition: u = 0, v = 1, dp/dn = 0
    u[:, 0, :] = 0
    v[:, 0, :] = 1
    p[:, 0, :] = p[:, 1, :]

    # Top (outflow) boundary condition: du/deta = 0, dv/deta = 0, p = 0
    p[:, -1, :] = 0
    u[:, -1, :] = u[:, -2, :]
    v[:, -1, :] = v[:, -2, :]

    # Left and right (wall) boundary conditions: u = 0, v = 0, dp/dxi = 0
    u[:, :, 0] = 0
    u[:, : , -1] = 0
    v[:, :, 0] = 0
    v[:, :, -1] = 0
    p[:, :, 0] = p[:, :, 1]
    p[:, :, -1] = p[:, :, -2]

    return torch.stack((u, v, p), dim=1)  # shape (1, 3, 77, 49)

## Physics Loss ##
####################################################################
def get_phys_derivs(f, h, coords):
    dfdxi = ddxi(f, h)
    dfdeta = ddeta(f, h)

    dfdx = coords['Jinv'] * (coords['dydeta'] * dfdxi - coords['dydxi'] * dfdeta)
    dfdy = coords['Jinv'] * (-coords['dxdeta'] * dfdxi + coords['dxdxi'] * dfdeta)

    return dfdx, dfdy


def physics_loss(pred, h, coords):
    u, v, p = pred[:, 0:1, :, :], pred[:, 1:2, :, :], pred[:, 2:3, :, :]

    # First derivatives
    dudx, dudy = get_phys_derivs(u, h, coords)
    dvdx, dvdy = get_phys_derivs(v, h, coords)
    dpdx, dpdy = get_phys_derivs(p, h, coords)


    d2udx2, _ = get_phys_derivs(dudx, h, coords) # x-derivative of dudx
    _, d2udy2 = get_phys_derivs(dudy, h, coords) # y-derivative of dudy
    
    d2vdx2, _ = get_phys_derivs(dvdx, h, coords) # x-derivative of dvdx
    _, d2vdy2 = get_phys_derivs(dvdy, h, coords) # y-derivative of dvdy

    # Residuals
    eq1 = (dudx + dvdy)
    eq2 = u*dudx + v*dudy + dpdx - 0.01*(d2udx2 + d2udy2)
    eq3 = u*dvdx + v*dvdy + dpdy - 0.01*(d2vdx2 + d2vdy2)



    return 100 * torch.mean(eq1**2) + torch.mean(eq2**2) + torch.mean(eq3**2)

## Training Loop ##
####################################################################
def train(model, optimizer, epochs, lr_input, coords, h):

    model.train()
    loss_vals = []

    for i in range(epochs):
        optimizer.zero_grad()

        out = model(lr_input)
        out = apply_boundary_conditions(out)

        loss = physics_loss(out, h, coords)

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_vals.append(loss.item())

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item()}")

    return loss_vals, out

## Main ##
####################################################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Prepare the data
    lr_input, Jinv, dxdxi, dxdeta, dydxi, dydeta = prepare_data(lfdata, hfdata, device)
    coords = {'Jinv': Jinv, 'dxdxi': dxdxi, 'dxdeta': dxdeta, 'dydxi': dydxi, 'dydeta': dydeta}
    h = .01


    # Initialize the model, loss function, and optimizer
    model = SRCNN().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    # Inside your train loop, after optimizer.step():
    scheduler.step()
    epochs = 5000
    
    # Train and catch the final high-res output
    loss_history, final_tensor = train(model, optimizer, epochs, lr_input, coords, h)

    # Plot using the tensor (the second item in the return)
    u_final = final_tensor[0, 0, :, :].detach().cpu().numpy()
    v_final = final_tensor[0, 1, :, :].detach().cpu().numpy()

    plt.plot(range(epochs), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Physics Loss")
    plt.title("Training Loss History")
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(hfx, hfy, np.sqrt(u_final**2 + v_final**2), cmap=cm.coolwarm, vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.title("Predicted Velocity Magnitude")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()