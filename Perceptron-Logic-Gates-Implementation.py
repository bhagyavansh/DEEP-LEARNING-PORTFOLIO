# Importing necessary libraries
import numpy as np                     # Helps us do fast math with arrays
import matplotlib.pyplot as plt        # Helps us make graphs and plots
import joblib                          # Helps us save/load models
import pandas as pd                    # Helps us handle data in table format

plt.style.use("fivethirtyeight")       # Makes the plot look stylish (not required here)

# Creating our Perceptron class
class Perceptron:
    def __init__(self, eta, epochs):             # eta is the learning rate, epochs is how many times to learn
        self.weights = np.random.randn(3) * 1e-4 # Start with 3 small random weights (2 for inputs + 1 for bias)
        print(f"Initial weights: {self.weights}")
        self.eta = eta                           # How fast the model learns (0.5 is okay)
        self.epochs = epochs                     # Number of training cycles (e.g., 10)

    def activationFunction(self, inputs, weights):   # Activation function decides output
        z = np.dot(inputs, weights)                 # Multiply inputs with weights and add them
        print("Dot product of inputs and weights:", z)
        return np.where(z > 0, 1, 0)                # If result > 0 → output is 1, else 0

    def fit(self, X, y):                            # This function trains the model
        self.X = X                                  # Store input features
        self.y = y                                  # Store actual output values
        X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]  # Add a column of -1 for bias
        print(f"X with bias: \n{X_with_bias}")

        for epoch in range(self.epochs):            # Repeat for number of epochs
            print(f"\nEpoch: {epoch}")
            y_hat = self.activationFunction(X_with_bias, self.weights)  # Predict output
            print(f"Predicted values: {y_hat}")
            error = self.y - y_hat                                 # Calculate difference from actual output
            print(f"Error: {error}")
            self.weights = self.weights + self.eta * np.dot(X_with_bias.T, error)  # Update weights
            print(f"Updated weights: {self.weights}")
            print("#############")

    def predict(self, X):                           # This function makes prediction
        X_with_bias = np.c_[X, -np.ones((len(X), 1))]  # Add bias column
        return self.activationFunction(X_with_bias, self.weights)  # Return 0 or 1


# AND Logic Gate
data = {"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1], "y": [0, 0, 0, 1]}  # AND gate truth table
AND = pd.DataFrame(data)                                           # Create table from data
print(AND)

X = AND.drop("y", axis=1)  # Get inputs only (x1, x2)
y = AND["y"]               # Get expected outputs (y)
y = y.to_frame()           # Convert to DataFrame if needed

model = Perceptron(eta=0.5, epochs=10)  # Create Perceptron model with learning rate and epochs
model.fit(X.values, y.values.ravel())  # Train model

# OR Logic Gate
data = {"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1], "y": [0, 1, 1, 1]}  # OR gate truth table
OR = pd.DataFrame(data)
print(OR)

X = OR.drop("y", axis=1)   # Get inputs
y = OR["y"]
y = y.to_frame()

model = Perceptron(eta=0.5, epochs=10)  # New model for OR gate
model.fit(X.values, y.values.ravel())

# XOR Logic Gate
data = {"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1], "y": [0, 1, 1, 0]}  # XOR gate truth table
XOR = pd.DataFrame(data)
print(XOR)

X = XOR.drop("y", axis=1)  # Inputs
y = XOR["y"]
y = y.to_frame()

model = Perceptron(eta=0.5, epochs=50)  # More epochs to try learning XOR (won’t fully succeed)
model.fit(X.values, y.values.ravel())

import os

# Create folder to store model
dir = "Perceptron_model"
os.makedirs(dir, exist_ok=True)  # Create the directory if it doesn't exist

# Save the trained model
filename = os.path.join(dir, 'AND_model.model')
joblib.dump(model, filename)

# Load model from file
loaded_model = joblib.load(filename)

# Predict using the loaded model
result = loaded_model.predict(X.values)
print("Prediction using loaded model:", result)
