# Solving Differential Equations with Neural Networks

The loss function of a Neural Network is usually described by some property including the predicted values of a model and the true values of the model, for example: 

$$ 
\text{loss} = (y_{\text{true}} - y_{\text{predicted}})^2 
$$

The loss function is something that we want to minimize to get an optimal model, i.e. \(\text{loss} \to 0\).

Differential equations, like the ODE: 

\[ 
y'(x) = y(x) 
\]

with condition 

\[ 
y(x = 0) = 1 
\]

can be put in the form 

\[ 
y'(x) - y(x) = 0, 
\]

i.e. the right-hand side of the equation can be set to zero. Here, \(y'(x)\) refers to the derivative of \(y\) with respect to \(x\).

We can approximate \(y(x)\) by employing an Artificial Neural Network so that 

$$ 
\text{ANN}(x) \approx y(x) 
$$

Instead of training the Neural Network where the loss function is defined by the difference between a true value and a predicted value, we train it by defining the loss function as the square of the differential equation and the square of the condition, i.e.

$$ 
\text{loss} = (ANN'(x) - ANN(x))^2 + (ANN(x = 0) - 1)^2. 
$$

Since we can define the structure of the Neural Network, we can find the derivative of the network with respect to some input \(x\), and we can also find the gradient of the loss function with respect to the internal parameters.

One of the big advantages of this method is that we can generate an arbitrarily large amount of data, which is one of the key factors for successfully optimizing any machine learning model.

This repository contains examples of how some differential equations are solved using this method, along with its python code.
