import unittest

import sys

from src import network as nn


class Test(unittest.TestCase):
    def test_can_create_a_neuron(self):
        neuron = nn.Neuron(5)
        print(neuron)
        self.assertEqual(len(neuron.weights), 5)

    def test_can_create_a_network(self):
        network = nn.NeuralNetwork([1,2,1])
        print(network)

        self.assertEqual(len(network.layers), 2)

        self.assertEqual(len(network.layers[0]), 2)
        self.assertEqual(len(network.layers[1]), 1)


    def test_can_feedforward(self):
        network = nn.NeuralNetwork([1, 2, 1])
        print(network.feedforward(5))


if __name__ == '__main__':
    unittest.main()
