import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy
from torchdiffeq import odeint
import pandas as pd

# odeint(func, y0, t)

class NeuODE(nn.Module):
    def __init__(self, width, hlayers):
        super(NeuODE, self).__init__()

        self.activation = nn.Tanh()

        layers = []
        layers.append(nn.Linear(4, width))
        layers.append(self.activation)
        for _ in range(hlayers-1):
            layers.append(nn.Linear(width, width))
            layers.append(self.activation)
        layers.append(nn.Linear(width, 4))

        self.network = nn.Sequential(*layers)

    def odefunc(self, t, y): # Output dydt
        return self.network(y)

    def forward(self, y0, tsteps):
        return odeint(
            self.odefunc, y0, tsteps,
            method="rk4",      # fixed step = much faster
            options={"step_size": 1.0}  # one step per month
        )


def data():
   
    # Read the data with pandas and concatenate train and test sets
    df1 = pd.read_csv("DailyDelhiClimateTrain.csv", parse_dates=["date"])
    df2 = pd.read_csv("DailyDelhiClimateTest.csv", parse_dates=["date"])
    df = pd.concat([df1, df2], ignore_index=True)

    # Save the index for plotting later
    df = df.set_index("date")

    # Take the mean of each month so the data is monthly
    monthly_df = df.resample("M").mean()

    # Normalize the data
    mean_vals = monthly_df.mean()
    std_vals = monthly_df.std()
    norm_df = (monthly_df - mean_vals) / std_vals

    # Split into train and test sets
    train_df = norm_df.iloc[:20]
    test_df = norm_df.iloc[20:]

    # Save dates for plotting later
    train_dates = train_df.index
    test_dates  = test_df.index

    # Save full forecast for plotting later
    full_dates = norm_df.index
    full_data  = torch.tensor(norm_df.values, dtype=torch.float64)

    train_data = torch.tensor(train_df.values, dtype=torch.float64)
    test_data  = torch.tensor(test_df.values, dtype=torch.float64)

    return train_data, test_data, train_dates, test_dates, mean_vals, std_vals, full_dates, full_data

def train_stage(y_train, model, optimizer, lossfn, epochs):
    model.train()
    stage_losses = []

    t = torch.arange(y_train.shape[0], device=y_train.device, dtype=y_train.dtype)


    for _ in range(epochs):
        optimizer.zero_grad()
        yhat = model(y_train[0, :], t) # Size: (T,4)
        loss = lossfn(yhat, y_train)
        loss.backward()
        optimizer.step()
        stage_losses.append(loss.item())

    return stage_losses

def test(y_test, t_test, model, lossfn):

    model.eval()
    with torch.no_grad():
        yhat = model(y_test[0, :], t_test)
        loss = lossfn(yhat, y_test)

    return loss.item()

def denormalize(y_norm, mean_vals, std_vals):
    # y_norm: torch tensor (T,4)
    mu = torch.tensor(mean_vals.values, dtype=y_norm.dtype, device=y_norm.device)
    sd = torch.tensor(std_vals.values, dtype=y_norm.dtype, device=y_norm.device)
    return y_norm * sd + mu

def forecast_all_months(model, y_all):
    """
    y_all: (T,4) normalized data across ALL months
    Forecast is produced using only y0 = first observation.
    """
    T = y_all.shape[0]
    t_all = torch.arange(T, device=y_all.device, dtype=y_all.dtype)
    y0 = y_all[0, :]  # (4,)
    with torch.no_grad():
        yhat_all = model(y0, t_all)  # (T,4)
    return yhat_all


## Once again had ChatGPT write the plot for me so it matched the blog post plotting system. ##
def plot_forecast(full_dates, y_obs_real, y_pred_real, split_idx=20):
    if torch.is_tensor(y_obs_real):  y_obs_real  = y_obs_real.detach().cpu().numpy()
    if torch.is_tensor(y_pred_real): y_pred_real = y_pred_real.detach().cpu().numpy()

    assert y_obs_real.ndim == 2 and y_obs_real.shape[1] == 4
    assert y_pred_real.ndim == 2 and y_pred_real.shape[1] == 4
    assert len(full_dates) == y_obs_real.shape[0] == y_pred_real.shape[0]

    names   = ["Mean temperature", "Humidity", "Wind speed", "Mean pressure"]
    ylabels = ["Celsius", "%", "km/h", "hPa"]
    colors  = ["#3b9de3", "#ca660f", "#288528", "#a876d7"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)

    for i, ax in enumerate(axes):
        ax.plot(full_dates, y_pred_real[:, i], linestyle="--", linewidth=2, color=colors[i], label="Forecast")
        ax.scatter(full_dates, y_obs_real[:, i], s=25, color=colors[i], edgecolor="k", linewidth=0.5, label="Observations")
        ax.axvline(full_dates[split_idx-1], color="k", linewidth=1, alpha=0.3)
        ax.set_title(names[i])
        ax.set_ylabel(ylabels[i])
        ax.grid(True, alpha=0.2)

    axes[0].legend()
    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    print("device:", device)

    # Get the data
    train_data, test_data, train_dates, test_dates, mean_vals, std_vals, full_dates, full_data = data()

    train_data = train_data.to(device=device, dtype=dtype)
    test_data  = test_data.to(device=device, dtype=dtype)
    full_data  = full_data.to(device=device, dtype=dtype)

    # Call the model, loss function, and optimizer
    model = NeuODE(width=20, hlayers=3).to(device=device, dtype=dtype)
    lossfn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Set up the stages of training
    stages = [4, 8, 12, 16, 20]
    epochs_per_stage = {4: 150, 8: 150, 12: 200, 16: 200, 20: 250}

    all_losses = []

    # Train in stages
    for k in stages:
        yk = train_data[:k, :]
        dk = train_dates[:k]

        stage_losses = train_stage(yk, model, optimizer, lossfn, epochs_per_stage[k])
        all_losses.extend(stage_losses)

        # predict over the same window for plotting
        with torch.no_grad():
            t = torch.arange(k, device=train_data.device, dtype=train_data.dtype)
            yhat = model(yk[0, :], t)  # (k,4)

    # plot_stage(dk, yk, yhat, stage_losses, title=f"Stage: first {k} months")
    # Observations across ALL months
    y_obs_real = denormalize(full_data, mean_vals, std_vals)

    yhat_all = forecast_all_months(model, full_data)         # (T,4)
    y_pred_real = denormalize(yhat_all, mean_vals, std_vals)

    plot_forecast(full_dates, y_obs_real, y_pred_real, split_idx=20)

    plt.plot(range(len(all_losses)), all_losses)
    plt.title("Training Loss Across All Stages")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)   
    plt.show()