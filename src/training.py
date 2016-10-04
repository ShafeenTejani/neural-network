import numpy as np
import random
import utils

class QuadraticCost:

    @staticmethod
    def cost(actual, target):
        return 0.5*np.linalg.norm(actual-target)**2

    @staticmethod
    def output_error(weighted_sum_of_inputs, activations, expected):
        return (activations - expected) * utils.sigmoid_prime(weighted_sum_of_inputs)


def stochastic_gradient_descent(network, training_data, epochs, mini_batch_size,
    learning_rate, test_data=None, cost=QuadraticCost):

    if test_data: num_test_data = len(test_data)

    for epoch in xrange(epochs):
        random.shuffle(training_data)
        mini_batches = get_mini_batches(training_data, mini_batch_size)

        for mini_batch in mini_batches:
            update_weights_for_batch(network, mini_batch, learning_rate, cost)

        if test_data:
            print "Epoch {0}: {1} / {2}".format(
                epoch, network.evaluate(test_data), num_test_data)
        else:
            print "Epock {0} complete".format(epoch)

def update_weights_for_batch(network, mini_batch, learning_rate, cost):
    bias_diffs = [np.zeros(b.shape) for b in network.biases]
    weight_diffs = [np.zeros(w.shape) for w in network.weights]

    for x, y in mini_batch:
        (delta_weight_diffs, delta_bias_diffs) = backpropagate(network, x, y, cost)
        bias_diffs = [bd + dbd for bd, dbd in zip(bias_diffs, delta_bias_diffs)]
        weight_diffs = [wd + dwd for wd, dwd in zip(weight_diffs, delta_weight_diffs)]

    update_weights(network, weight_diffs, learning_rate / len(mini_batch))
    update_biases(network, bias_diffs, learning_rate / len(mini_batch))


def update_weights(network, weight_differentials, learning_rate):
    network.weights = [w - (learning_rate)*dw
        for w, dw in zip(network.weights, weight_differentials)]

def update_biases(network, bias_differentials, learning_rate):
    network.biases = [b - (learning_rate)*db
        for b, db in zip(network.biases, bias_differentials)]

def backpropagate(network, x, y, cost):
    (activations, outputs) = network.neuron_activations(x)
    activations = [x] + activations
    #output layer
    layer_errors = cost.output_error(outputs[-1], activations[-1], y)
    weight_differentials = [np.dot(layer_errors, activations[-2].transpose())]
    bias_differentials = [layer_errors]
    #hidden layers
    for layer in xrange(2, network.num_layers):
        layer_output = outputs[-layer]
        layer_sigmoid_prime = utils.sigmoid_prime(layer_output)
        prev_layer_weights = network.weights[-layer+1]
        prev_layer_errors = layer_errors
        layer_errors = hidden_layer_errors(prev_layer_weights, prev_layer_errors, layer_sigmoid_prime)
        weight_differentials.append(np.dot(layer_errors, activations[-layer-1].transpose()))
        bias_differentials.append(layer_errors)

    return (reversed(weight_differentials), reversed(bias_differentials))


def hidden_layer_errors(prev_layer_weights, prev_layer_errors, layer_sigmoid_prime):
    return np.dot(prev_layer_weights.transpose(), prev_layer_errors) * layer_sigmoid_prime

def get_mini_batches(training_data, mini_batch_size):
    return [ training_data[k:k + mini_batch_size] for k in xrange(0, len(training_data), mini_batch_size) ]
