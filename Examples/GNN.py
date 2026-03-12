import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import matplotlib.pyplot as plt

class GCN(nn.Module):

    def __init__(self, hidden, dataset):
        super().__init__()

        self.conv1 = GCNConv(dataset.num_features, hidden)
        self.conv2 = GCNConv(hidden, dataset.num_classes)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train(graph, model, loss_fn, optimizer):
    model.train()
    optimizer.zero_grad()

    yhat = model(graph.x, graph.edge_index)
    loss = loss_fn(yhat[graph.train_mask], graph.y[graph.train_mask])

    loss.backward()
    optimizer.step()

    return loss.item()

def test(graph, model):
    model.eval()
    with torch.no_grad():
        yhat = model(graph.x, graph.edge_index)
        

if __name__ == "__main__":

    dataset = Planetoid(root='planetoid', name='Cora', transform=NormalizeFeatures())

    graph = dataset[0]

    model = GCN(hidden=16, dataset=dataset)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    epochs = 200
    train_losses = []
    for epoch in range(epochs):
        loss = train(graph, model, F.cross_entropy, optimizer)
        train_losses.append(loss)
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Progress")
    plt.show()