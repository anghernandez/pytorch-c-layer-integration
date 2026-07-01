import torch
import numpy as np
# Importa el módulo compilado con pybind11.
import dense_pybind

torch.manual_seed(0)

x = np.array([
    [0.5, -1.0, 2.0],
    [1.0,  0.0, -0.5]
], dtype=np.float32)

weights = (torch.randn(2, 3) * 0.01).numpy().astype(np.float32)
bias = torch.zeros(2).numpy().astype(np.float32)

y = dense_pybind.dense_tanh_forward(x, weights, bias)

print("Weights:")
print(weights)

print("\nBias:")
print(bias)

print("\nSalida pybind11:")
print(y)