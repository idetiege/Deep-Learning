import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
import matplotlib.pyplot as plt

class Encode(nn.Module):
    def __init__(self):
        super(Encode, self).__init__()

        self.activation = nn.ReLU()

        self.network = nn.Sequential(
            nn.Linear(7, 30),
            self.activation,
            nn.Linear(30, 30),
            self.activation,
            nn.Linear(30, 60),
        )

    def forward(self, x):
        return self.network(x)

class Network(nn.Module):
    def __init__(self, K0):
        super(Network, self).__init__()

        self.K = nn.Sequential(
            nn.Linear(60, 60, bias=False), 
        )

    def forward(self, x):
        return self.K(x)

class Decode(nn.Module):
    def __init__(self):
        super(Decode, self).__init__()

        self.activation = nn.ReLU()

        self.network = nn.Sequential(
            nn.Linear(60, 30),
            self.activation,
            nn.Linear(30, 30),
            self.activation,
            nn.Linear(30, 7)
        )

    def forward(self, x):
        return self.network(x)

def data():
    ntraj = 2148  # number of trajectories
    nt = 50  # number of time steps
    ny = 7  # number of states

    tvec = np.linspace(0, 350, nt)
    Y = np.loadtxt('kdata.txt').reshape(ntraj, nt, ny)
    Ytrain = Y[:2048, :, :]  # 2048 training trajectories
    Ytest = Y[2048:, :, :]  # 100 testing trajectoreis

    Ytrain_torch = torch.from_numpy(Ytrain).float() # Size: (2048, 50, 7)

    y_t = Ytrain_torch[:, :-1, :] # (N_traj, N_t-1, 7)
    y_t1 = Ytrain_torch[:, 1:, :] # (N_traj, N_t-1, 7)

    y_t_flat = y_t.reshape(-1, 7) # (N_traj*(N_t-1), 7)
    y_t1_flat = y_t1.reshape(-1, 7) # (N_traj*(N_t-1), 7)

    first_traj = Ytrain[:3, :, :]
    second_traj = Ytrain[3:6, :, :]
    third_traj = Ytrain[6:9, :, :]


    return tvec, Ytrain, Ytest, y_t_flat, y_t1_flat, first_traj, second_traj, third_traj

def Loss_recon(loss_fn, encoder, decoder, y_batch):

    z = encoder(y_batch)
    Y_hat = decoder(z)

    L = loss_fn(Y_hat, y_batch)
    return L

def Loss_linear(loss_fn,encoder, K, y_t, y_t1):
    z_t = encoder(y_t)
    z_t1 = encoder(y_t1)
    z_t1_hat =  K(z_t)

    L = loss_fn(z_t1, z_t1_hat)
    return L

def Loss_pred(loss_fn, encoder, decoder, network, y_t_flat, y_t1_flat):
    
    z_t = encoder(y_t_flat)
    z_t1_hat = network(z_t)
    y_t1_hat = decoder(z_t1_hat)
    
    L = loss_fn(y_t1_hat, y_t1_flat)
    return L

def train(model, optimizer, loss_fn, encoder, decoder, network, y_t_flat, y_t1_flat, num_epochs):
    loss = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        loss_pred = Loss_pred(loss_fn, encoder, decoder, network, y_t_flat, y_t1_flat)
        loss_linear = Loss_linear(loss_fn, encoder, network, y_t_flat, y_t1_flat)
        loss_recon = Loss_recon(loss_fn, encoder, decoder, y_t_flat)
        loss = loss_pred + loss_linear + loss_recon

        loss.backward()
        optimizer.step()

        loss.append(loss)

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        return loss
        

def rollout_prediction(encoder, decoder, network, traj, nt=50):
    """
    traj : shape (50, 7) for ONE trajectory
    nt   : number of timesteps to predict
    Returns predicted states with shape (nt, 7)
    """

    # Start from the *true* first state
    y0 = torch.tensor(traj[0], dtype=torch.float).unsqueeze(0)   # shape (1, 7)

    # Encode initial state
    z = encoder(y0)

    preds = []
    for t in range(nt):
        # Decode latent vector back to physical state
        y_hat = decoder(z)
        preds.append(y_hat.squeeze(0).detach().numpy())

        # Advance latent state using learned linear map
        z = network(z)

    return np.array(preds)   # shape (nt, 7)


def test(model, loss_fn, encoder, decoder, network, y_t_flat, y_t1_flat):
    model.eval()
    with torch.no_grad():
        loss_pred = Loss_pred(loss_fn, encoder, decoder, network, y_t_flat, y_t1_flat)
        loss_linear = Loss_linear(loss_fn, encoder, network, y_t_flat, y_t1_flat)
        loss_recon = Loss_recon(loss_fn, encoder, decoder, y_t_flat)
        loss_val = loss_pred + loss_linear + loss_recon
        print(f'Test Loss: {loss_val.item():.4f}')
    return [loss_val.item()]

if __name__ == "__main__":

    tvec, Ytrain, Ytest, y_t_flat, y_t1_flat, first_traj, second_traj, third_traj = data()

    lr = 1e-3
    encoder = Encode()
    decoder = Decode()
    network = Network(encoder)
    optimizer = AdamW(list(encoder.parameters()) + 
                      list(decoder.parameters()) + 
                      list(network.parameters()), 
                      lr=lr)
    
    loss_fn = nn.MSELoss()
    num_epochs = 1000
    train_loss = train(network, optimizer, loss_fn, encoder, decoder, network, y_t_flat, y_t1_flat, num_epochs=num_epochs)
    test_loss = test(network, loss_fn, encoder, decoder, network, y_t_flat, y_t1_flat)

    pred1 = rollout_prediction(encoder, decoder, network, first_traj[0])
    pred2 = rollout_prediction(encoder, decoder, network, second_traj[0])
    pred3 = rollout_prediction(encoder, decoder, network, third_traj[0])

    plt.figure(figsize=(10,6))

    # True
    plt.plot(tvec, first_traj[0, :, 0], 'k--', label='True Traj 1')
    plt.plot(tvec, second_traj[0, :, 0], 'k-.', label='True Traj 2')
    plt.plot(tvec, third_traj[0, :, 0], 'k:', label='True Traj 3')

    # Predicted
    plt.plot(tvec, pred1[:, 0], label='Pred 1')
    plt.plot(tvec, pred2[:, 0], label='Pred 2')
    plt.plot(tvec, pred3[:, 0], label='Pred 3')
    plt.xlabel("Time")
    plt.ylabel("State 0")
    plt.legend()
    plt.title("True vs Predicted Trajectories")
    plt.savefig("trajectories.png")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(range(num_epochs), train_loss, label='Train Loss')
    plt.plot(num_epochs-1, test_loss[0], 'ro', label='Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("testing_loss.png")
    plt.show()

