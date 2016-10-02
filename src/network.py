import random

import numpy as np
import json

import network_utils as utils

class Neuron:
    def __init__(self, number_of_inputs):
        self.weights = np.random.randn(number_of_inputs)
        self.bias = np.random.rand()

    def __str__(self):
        return json.dumps(self.as_dict())

    def output(self, inputs):
        return utils.sigmoid(np.dot(self.weights, inputs) + self.bias)

    def as_dict(self):
        return {'weights': self.weights.tolist(), 'bias': self.bias}


class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.number_of_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.layers = [self._create_random_neurons(i, layer_sizes) for i in range(1, self.number_of_layers)]

    def feedforward(self, input):
        output_of_prev_layer = input
        for layer in self.layers:
            output_of_prev_layer = [neuron.output(output_of_prev_layer) for neuron in layer]
        return output_of_prev_layer

    def as_dict(self):
        dictionary = {}
        for index, layer in enumerate(self.layers):
            dictionary['layer-' + str(index + 1)] = [neuron.as_dict() for neuron in layer]
        return dictionary


    def _create_random_neurons(self, index, layer_sizes):
        number_of_inputs = layer_sizes[index-1]
        return [self._create_random_neuron(number_of_inputs) for i in range(0,layer_sizes[index])]

    def _create_random_neuron(self, number_of_inputs):
        return Neuron(number_of_inputs);

    def __str__(self):
        return json.dumps(self.as_dict())
