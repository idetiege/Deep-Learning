import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW

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

## Data Prep
####################################################################

# Define tensors for Jinv, dxdxi, dxdeta, dydxi, dydeta, hfx, hfy, lfx, lfy, lfu, lfv
def prepare_data():

    return


## Model Architecture
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

## Boundary Conditions
####################################################################
# Overwrite the network's predictions at the edges to satisfy the physics of the problem.
def apply_boundary_conditions():

    return

## Physics Loss
####################################################################
def physics_loss():

    return

## Training Loop
####################################################################
def train():

    return

## Main
####################################################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Prepare the data
    prepare_data()

    # Initialize the model, loss function, and optimizer
    model = SRCNN().to(device)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Train the model
    train()

    # Plot the results
