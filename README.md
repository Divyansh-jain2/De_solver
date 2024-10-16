# Solving Ordinary Differential Equations Using Neural Networks

## Author: Divyansh Jain (B23397)

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

This projecn libraries:t requires Python 3.x and the following Pytho
- `torch` (PyTorch)
- `numpy`
- `matplotlib` (for plotting results)

You can install the required libraries using:
```bash
pip install torch numpy matplotlib
