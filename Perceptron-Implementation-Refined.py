import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
plt.style.use("fivethirtyeight")

class Perceptron:
    def __init__(self, eta, epochs):
        self.weights = np.random.randn(3) * 1e-4
        print(f"self.weights: {self.weights}")
        self.eta = eta
        self.epochs = epochs

    def activationFunction(self, inputs, weights):
        z = np.dot(inputs, weights)
        print("Dot Product of input and weights: ", z)
        return np.where(z > 0, 1, 0)

    def fit(self, x, y):
        self.x = x
        self.y = y
        x_with_bias = np.c_[self.x, -np.ones((len(self.x), 1))]  # Concatenation
        print(f"X_with_bias: \n{x_with_bias}")

        for epoch in range(self.epochs):
            print(f"for epoch: {epoch}")
            y_hat = self.activationFunction(x_with_bias, self.weights)
            print(f"Predicted value: \n{y_hat}")
            error = self.y - y_hat
            print(f"error: \n{error}")
            self.weights = self.weights + self.eta * np.dot(x_with_bias.T, error)
            print(f"updated weights: \n{self.weights}")
            print("##########\n")

    def predict(self, x):
        x_with_bias = np.c_[x, -np.ones((len(x), 1))]  
        return self.activationFunction(x_with_bias, self.weights)


data = {"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1], "y": [0, 0, 0, 1]}
AND = pd.DataFrame(data)

x = AND[['x1', 'x2']]  
y = AND['y']  

model = Perceptron(eta=0.5, epochs=10)
model.fit(x, y)  

#OR Operation
data = {"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1], "y": [0, 1, 1, 1]}

OR = pd.DataFrame(data)
OR

x = AND.drop("y", axis = 1)
x