from pysr import PySRRegressor
import numpy as np

data = np.loadtxt('subset.csv', delimiter = ',')

y = data[:, 0]
X = data[:, 1:]

model = PySRRegressor(
    niterations=1000,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log", "sin", "cos"],
    populations=20,
    population_size=100,
    model_selection="best",
    maxsize=20,
)

model.fit(X, y)

print(model)