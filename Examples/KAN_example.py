import kan
import torch
import matplotlib.pyplot as plt

f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)

dataset = kan.utils.create_dataset(f, n_var = 2)

# 2 inputs, 5 hidden, 1 output, grid size 3, cubic spline
model = kan.KAN(width = [2, 5, 1], grid = 3, k=3, seed=42)

model.fit(dataset, opt='LBFGS', steps = 50, lamb=0.001)
model.plot()

model = model.prune()
model.plot()
plt.show()

model = model.refine(10)
model.fit(dataset, opt ='LBFGS', steps = 50)

model.auto_symbolic()
model.fit(dataset, opt='LBFGS', steps = 100)

from kan.utils import ex_round
print(ex_round(model.symbolic_formula()[0][0], 4))