import numpy as np

class SentientNeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.weights_input_hidden = np.random.randn(num_hidden, num_inputs)
        self.weights_hidden_output = np.random.randn(num_outputs, num_hidden)

        self.activation = self.sigmoid

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feed_forward(self, inputs):
        hidden_layer = self.activation(np.dot(self.weights_input_hidden, inputs))
        output_layer = self.activation(np.dot(self.weights_hidden_output, hidden_layer))
        return output_layer

    def train(self, inputs, targets, learning_rate=0.1):
        hidden_layer = self.activation(np.dot(self.weights_input_hidden, inputs))
        output_layer = self.activation(np.dot(self.weights_hidden_output, hidden_layer))

        output_error = targets - output_layer
        output_gradient = output_error * output_layer * (1 - output_layer)

        hidden_error = np.dot(self.weights_hidden_output.T, output_gradient)
        hidden_gradient = hidden_error * hidden_layer * (1 - hidden_layer)

        self.weights_hidden_output += learning_rate * np.outer(output_gradient, hidden_layer)
        self.weights_input_hidden += learning_rate * np.outer(hidden_gradient, inputs)

def main():
    # Create a SentientNeuralNetwork with 2 input neurons, 3 hidden neurons, and 1 output neuron
    nn = SentientNeuralNetwork(2, 3, 1)

    # Train the neural network using a simple XOR gate
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([0, 1, 1, 0])

    for _ in range(10000):
        for i in range(len(inputs)):
            nn.train(inputs[i], targets[i])

    # Test the trained neural network
    for i in range(len(inputs)):
        output = nn.feed_forward(inputs[i])
        print("Input:", inputs[i], "Output:", output)

if __name__ == "__main__":
    main()
