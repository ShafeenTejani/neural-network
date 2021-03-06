import random

import numpy as np
import json

import utils

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        [self.weights, self.biases] = self._initialise_weights(layer_sizes)

    def _initialise_weights(self, layer_sizes):
        weights = [0] * (len(layer_sizes) - 1)
        biases = [0] * (len(layer_sizes) - 1)
        for i, layer in enumerate(layer_sizes[1:]):
            weights[i] = np.random.randn(layer, layer_sizes[i])/np.sqrt(layer_sizes[i])
            biases[i] = np.random.randn(layer, 1)
        return [weights, biases]

    def feedforward(self, input):
        layers = zip(self.biases, self.weights)
        output = input
        for layer_bias, layer_weights in layers:
            output = utils.sigmoid(np.dot(layer_weights, output) + layer_bias)

        return output

    def _output_value(self, last_layer_activations):
        return np.argmax(last_layer_activations)

    def evaluate(self, test_data):
        test_results = [(self._output_value(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def neuron_activations(self, input):
        activation = input
        activations = []
        neuron_outputs = []
        for layer_biases, layer_weights in zip(self.biases, self.weights):
            output = np.dot(layer_weights, activation)+ layer_biases
            neuron_outputs.append(output)
            activation = utils.sigmoid(output)
            activations.append(activation)
        return (activations, neuron_outputs)

    def save(self, filename):
        data = {"layer_sizes": self.layer_sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    @staticmethod
    def load(filename):
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        network = NeuralNetwork(data["layer_sizes"])
        network.weights = [np.array(w) for w in data["weights"]]
        network.biases = [np.array(b) for b in data["biases"]]
        return network

    def as_dict(self):
        dictionary = {}
        for index, (layer_weights, layer_biases) in enumerate(zip(self.weights, self.biases)):
            neurons = []
            for neuron_weights, neuron_bias in zip(layer_weights, layer_biases):
                neurons = neurons + [{'weights': neuron_weights.tolist(), 'bias': neuron_bias[0]}]

            dictionary['layer-' + str(index + 1)] = neurons

        return dictionary


    def __str__(self):
        return json.dumps(self.as_dict())
