import numpy as np
import math
import matplotlib.pyplot as plt

# -------- activation functions -------
def relu(z):
    return np.maximum(0, z)

def relu_back(xbar, z):
    return xbar * (z > 0)

identity = lambda z: z

identity_back = lambda xbar, z: xbar
# -------------------------------------------


# ---------- initialization -----------
def initialization(nin, nout):
    W = np.random.normal(0, 2 / (nin + nout), (nout, nin))
    b = np.zeros((nout, 1))
    return W, b
# -------------------------------------


# -------- loss functions -----------
def mse(yhat, y):
    return np.sum((y - yhat)**2) / y.shape[1]

def mse_back(yhat, y):
    return 2 * (yhat - y) / y.shape[1]
# -----------------------------------


# ------------- Layer ------------
class Layer:

    def __init__(self, nin, nout, activation=identity):
        W, b = initialization(nin, nout)
        self.W = W
        self.b = b
        self.activation = activation

        if activation == relu:
            self.activation_back = relu_back
        if activation == identity:
            self.activation_back = identity_back

        # initialize cache
        self.cache = {}

    def forward(self, X, train=True):
        Z = self.W @ X + self.b
        Xnew = self.activation(Z)

        # save cache
        if train:
            self.cache['X'] = X
            self.cache['Z'] = Z
        return Xnew

    def backward(self, Xnewbar):
        X = self.cache['X'] # (nin, ns)
        Z = self.cache['Z'] # (nout, ns)
        ns = X.shape[1]

        Zbar = self.activation_back(Xnewbar, Z)  # (nout, ns)
        
        self.Wbar = (Zbar @ X.T) # (nout, nin)
        self.bbar = np.sum(Zbar, axis=1, keepdims=True) # (nout, 1)

        Xbar = self.W.T @ Zbar  # (nin, ns)
        return Xbar


class Network:

    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss
        self.cache = {}

        if loss == mse:
            self.loss_back = mse_back

    def forward(self, X, y, train=True):

        A = X
        for layer in self.layers:
            A = layer.forward(A, train=train)

        yhat = A
        L = self.loss(yhat, y)

        if train:
            self.cache['X'] = X
            self.cache['y'] = y
            self.cache['yhat'] = yhat

        return L, yhat

    def backward(self):
        y = self.cache['y']
        yhat = self.cache['yhat']

        yhatbar = self.loss_back(yhat, y)

        Abar = yhatbar
        for layer in reversed(self.layers):
            Abar = layer.backward(Abar)


class GradientDescent:

    def __init__(self, alpha):
        self.alpha = alpha

    def step(self, network):
        for layer in network.layers:
            layer.W -= self.alpha * layer.Wbar
            layer.b -= self.alpha * layer.bbar


if __name__ == '__main__':

    # ---------- data preparation ----------------
    # Initialize lists for the numeric data and the string data
    numeric_data = []

    # Read the text file
    with open('auto-mpg.data', 'r') as file:
        for line in file:
            # Split the line into columns
            columns = line.strip().split()

            # Check if any of the first 8 columns contain '?'
            if '?' in columns[:8]:
                continue  # Skip this line if there's a missing value

            # Convert the first 8 columns to floats and append to numeric_data
            numeric_data.append([float(value) for value in columns[:8]])

    # Convert numeric_data to a numpy array for easier manipulation
    numeric_array = np.array(numeric_data)

    # Shuffle the numeric array and the corresponding string array
    nrows = numeric_array.shape[0]
    indices = np.arange(nrows)
    np.random.shuffle(indices)
    shuffled_numeric_array = numeric_array[indices]

    # Split into training (80%) and test (20%) sets
    split_index = int(0.8 * nrows)

    train_numeric = shuffled_numeric_array[:split_index]
    test_numeric = shuffled_numeric_array[split_index:]

    # separate inputs/outputs
    Xtrain = train_numeric[:, 1:]
    ytrain = train_numeric[:, 0]

    Xtest = test_numeric[:, 1:]
    ytest = test_numeric[:, 0]

    # normalize
    Xmean = np.mean(Xtrain, axis=0)
    Xstd = np.std(Xtrain, axis=0)
    ymean = np.mean(ytrain)
    ystd = np.std(ytrain)

    Xtrain = (Xtrain - Xmean) / Xstd
    Xtest = (Xtest - Xmean) / Xstd
    ytrain = (ytrain - ymean) / ystd
    ytest = (ytest - ymean) / ystd

    # reshape arrays (opposite order of pytorch, here we have nx x ns).
    # I found that to be more conveient with the way I did the math operations, but feel free to setup
    # however you like.
    Xtrain = Xtrain.T
    Xtest = Xtest.T
    ytrain = np.reshape(ytrain, (1, len(ytrain)))
    ytest = np.reshape(ytest, (1, len(ytest)))

    # ------------------------------------------------------------

    l1 = Layer(7, 32, relu)
    l2 = Layer(32, 20, relu)
    l3 = Layer(20, 1, identity)
    layers = [l1, l2, l3]
    network = Network(layers, mse)
    alpha = 0.1
    optimizer = GradientDescent(alpha)

    train_losses = []
    test_losses = []
    epochs = 800
    for i in range(epochs):
        # --- train ---
        Ltrain, _ = network.forward(Xtrain, ytrain, train=True)
        network.backward()
        optimizer.step(network)

        # --- test ---
        Ltest, _ = network.forward(Xtest, ytest, train=False)

        train_losses.append(Ltrain)
        test_losses.append(Ltest)

        if (i+1) % 25 == 0:
            print(f"epoch {i+1}/{epochs}  train={Ltrain:.4f}  test={Ltest:.4f}")


    # --- inference ----
    _, yhat = network.forward(Xtest, ytest, train=False)

    # unnormalize
    yhat = (yhat * ystd) + ymean
    ytest = (ytest * ystd) + ymean

    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Losses')
    plt.savefig('loss_plot.png')
    plt.legend()


    plt.figure()
    plt.plot(ytest.T, yhat.T, "o")
    plt.plot([10, 45], [10, 45], "--")
    plt.savefig('prediction_plot.png')
    print("avg error (mpg) =", np.mean(np.abs(yhat - ytest)))

    plt.show()