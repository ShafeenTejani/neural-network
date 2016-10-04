import unittest

import sys

from src import network as nn


class Test(unittest.TestCase):

    def test_can_create_a_network(self):
        network = nn.NeuralNetwork([1,2,1])
        print(network)

        self.assertEqual(len(network.weights), 2)
        self.assertEqual(len(network.weights[0]), 2)
        self.assertEqual(len(network.weights[1]), 1)


    def test_can_feedforward(self):
        network = nn.NeuralNetwork([1, 2, 1])
        print(network.feedforward(5))

    def test_neuron_activations(self):
        network = nn.NeuralNetwork([1, 2, 1])
        (activations, outputs) = network.neuron_activations(5)
        print "activations"
        print activations
        print "outputs"
        print outputs


if __name__ == '__main__':
    unittest.main()
