
# Solving Ordinary Differential Equations Using Neural Networks

## Author: Divyansh Jain

### Overview

This project demonstrates how to solve second-order ordinary differential equations (ODEs) using neural networks. The neural network is trained to approximate the solution to the differential equation by minimizing a custom loss function that encodes both the ODE and its boundary conditions.

### Problem Statement

We solve the following second-order differential equation:
\[
\frac{d^2y}{dx^2} + 2\frac{dy}{dx} - 3y = 0, \quad 0 \leq x \leq 1,
\]
with the boundary conditions:
\[
y(0) = 1, \quad y(1) = e^1.
\]

The exact analytical solution to this equation is \( y(x) = e^x \).

### Methodology

A simple feedforward neural network is used to approximate the solution of the ODE. Key aspects of the method are:
- **Network Structure**: A single hidden layer with 10 neurons and a sigmoid activation function.
- **Trial Solution**: The trial solution ensures that the boundary conditions are met by construction.
- **Loss Function**: The loss function includes both the residual of the ODE and penalties for the boundary conditions.
- **Optimizer**: The LBFGS optimizer is used due to its efficiency in solving optimization problems that require high precision.

### Installation

This project requires Python 3.x and the following Python libraries:
- `torch` (PyTorch)
- `numpy`
- `matplotlib` (for plotting results)

You can install the required libraries using:
```bash
pip install torch numpy matplotlib
```

### Running the Code

To solve the ODE using a neural network, follow these steps:

1. **Define the Neural Network Model**: The model is a fully connected network with one hidden layer and sigmoid activations.
2. **Define the Loss Function**: This includes the residual of the differential equation and the boundary conditions.
3. **Train the Model**: Use the LBFGS optimizer to minimize the loss and train the model.
4. **Evaluate the Results**: After training, compare the network's output with the exact solution \( y(x) = e^x \).

### Example Usage

```python
import torch
import torch.nn as nn

# Define the Neural Network
class Network2(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(1, 10)
        self.output_layer = nn.Linear(10, 1)

    def forward(self, x):
        layer_out = torch.sigmoid(self.hidden_layer(x))
        output = self.output_layer(layer_out)
        return output

# Define the loss function
def loss(x, model):
    x.requires_grad = True
    y = model(x)
    
    # Calculate derivatives
    dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    d2y_dx2 = torch.autograd.grad(dy_dx.sum(), x, create_graph=True)[0]
    
    # ODE residual
    ode_residual = d2y_dx2 + 2 * dy_dx - 3 * y
    
    # Boundary conditions
    boundary_loss = 0.5 * (y[0, 0] - 1.)**2 + 0.5 * (y[-1, 0] - torch.exp(torch.tensor(1.0)))**2
    
    return torch.mean(ode_residual**2) + boundary_loss

# Train the model
# Use LBFGS optimizer, define training loop and closure function...
```

### Results

The network successfully approximates the solution \( y(x) = e^x \) with minimal error. The graph comparing the neural network's output to the exact solution is included in the report.

### References

- PyTorch documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

### License

This project is open-source and is distributed under the MIT License.
