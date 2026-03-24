import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint

seed = 42
torch.manual_seed(seed)

# Neural ODE def. Same as trhe homework mostly
class NeuODE(nn.Module):
    def __init__(self, width=32, hlayers=2):
        super(NeuODE, self).__init__()

        self.activation = nn.Tanh()

        layers = []
        layers.append(nn.Linear(2, width))
        layers.append(self.activation)
        for _ in range(hlayers-1):
            layers.append(nn.Linear(width, width))
            layers.append(self.activation)
        layers.append(nn.Linear(width, 1))

        self.network = nn.Sequential(*layers)

    def odefunc(self, t, y): # Output dvdt
        x = y[:,0:1]
        v = y[:, 1:2]

        # ODE structure
        dvdt = torch.cat([x, v], dim=1)
        dvdt = self.network(dvdt)
        f = torch.cat([v, dvdt], dim=1)
        return f

    def forward(self, y0, tsteps):
            return odeint(
                self.odefunc, y0, tsteps)


def data():
    data = np.loadtxt("timeposition.txt")
    t = torch.tensor(data[:,0], dtype=torch.float32) # Size ncol x 1
    x = torch.tensor(data[:,1:6], dtype=torch.float32) # size ncol x 5

    # I dont think i need to normalize initially. We'll find out.

    # Split into train and test sets
    tTrain = t
    xTrain = x
    # tTest = t[80:] 
    # xTest = x[80:]

    xInitial = xTrain[0].unsqueeze(1)
    vInitial = torch.zeros_like(xInitial)
    y0 = torch.cat([xInitial, vInitial], dim=1)

    # im not going to batch becuse there's only 100 data points.

    # print("t shape:", t.shape)
    # print("x shape:", x.shape)
    return tTrain, xTrain, y0, t


# Loss function
def loss(pred, target, loss_fn):
    return loss_fn(pred, target)

# Training and testing
def train(model, tTrain, xTrain, states, optimizer, epochs):
    model.train()
    train_losses = []

    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(states, tTrain)
  

        pred_pos = pred[:, :, 0]
        l = nn.MSELoss()(pred_pos, xTrain)
        l.backward()
        optimizer.step()
        train_losses.append(l.item())
        if _ % 10 == 0:
            print(f"Epoch {_+1}/{epochs}, Loss: {l.item():.4f}")
    return train_losses

# def test(y_test, t_test, model, lossfn):

#     model.eval()
#     with torch.no_grad():
#         yhat = model(y_test[0, :], t_test)
#         loss = lossfn(yhat, y_test)
            
#     return loss.item()

if __name__ == "__main__":
    epochs = 200
    tTrain, xTrain, y0, tFull = data()

    model = NeuODE(width=32, hlayers=2)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # Run the training
    losses = train(model, tTrain, xTrain, y0, optimizer, epochs)

    model.eval()
    t = torch.linspace(0, tFull[-1], 200)

    # First trajectory x and v values
    x_first = 1
    v0_first = 0
    y0_new = torch.tensor([[x_first, v0_first]], dtype=torch.float32)

    # Model output for this initial cond
    with torch.no_grad():
        output = model(y0_new, t)
    
    # predicted velocity
    vel = output[:, 0, 1]

    # Plot training loss for hyperparameter tuning: 75 seems good but ill just run the 200 like Dr. Ning
    # plt.plot(range(len(losses)), losses)
    # plt.title("Training Loss")
    # plt.yscale("log")
    # plt.xlabel("Epoch")
    # plt.ylabel("MSE Loss")
    # plt.show()


    # Plotting result
    plt.plot(t.numpy(), vel.numpy(), label='Trajectory 1')
    plt.xlabel("Time")
    plt.ylabel("Velocity (v)")
    plt.title("Predicted Velocity (Initial Cond: x=1, v=0)")
    plt.savefig('predicted_velocity.png')
    plt.legend()
    plt.show()