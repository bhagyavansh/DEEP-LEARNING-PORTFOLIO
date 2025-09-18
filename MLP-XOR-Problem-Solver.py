import numpy as np

class MLP:
    def __init__(self, layers, learning_rate=0.1):
        """
        Initializes a multi-layer perceptron.
        :param layers: List containing the number of neurons in each----
        :param learning_rate: Learning weight of rate updates.
        """

        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        #Initialize weights and biases for each layer
        for i in range(len(layers) - 1):
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * 0.1)
            self.biases.append(np.random.randn(layers[i+1]) * 0.1)
    def    relu(self, x):
        return np.maximum(0, x)

    def    relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1-x)

    def forward(self, x):
        """
        Forward pass through the network.
        param x:input data.
        return: Outputs of each layer.
        """

        activations = [x]
        for i in range(len(self.weights)-1):
            x = self.relu(np.dot(x, self.weights[i]) + self.biases[i])
            activations.append(x)

            #Output layer uses sigmoid for classification
            x = self.sigmoid(np.dot(x, self.weights[-1]) + self.biases[-1])
            activations.append(x)
            return activations

        def backward(self, activations, y):
            """
            Backward pass using backpropagation.
            :Param activations: Output from forward pass.
            :Param y: True labels.
            """

            deltas = [activations[-1] - y] #Error at output layer

            #Compute gradients for hiden layers

            for i in range(len(self.weights) -2, -1, -1):
                deltas.append(deltas[-1] @ self.weights[i + 1].T * self.relu_derivative(activations[i + 1]))
                
                deltas.reverse()
                #Weights and biases
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * np.outer(activations[i], deltas[i])
                    self.biases[i] -= self.learning_rate * deltas[i]

            def train(self, x,y,epochs = 10000):
                for epoch in range(epochs):
                    for i in range(len(x)):
                        activations = self.forward(x[i])
                        self.backward(activations, y[i])

            def predict(self, x):
                return self.forward(x)[-1]

#Example usage (XOR Problem)
x =  np.array([[0, 0], [0, 1], [1,0], [1, 1]]) #inputs
y = np.array([[0], [1], [1], [0]]) # XOR outputs

mlp = MLP(layers=[2, 4, 1], learning_rate=0.1)
mlp.train(x, y, epochs = 10000)

#Testing
for i in range(len(x)):
    print(f"Input: {x[i]}, Predicted Output: {mlp.predict(x[i])}")

