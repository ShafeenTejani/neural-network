import numpy as np

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def stochastic_gradient_descent(network, training_data, epochs, mini_batch_size,
    learning_rate, test_data=None):
        for epoch in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = get_mini_batches(training_data, mini_batch_size)

            for mini_batch in mini_batches:
                update_weights(network, mini_batch, learning_rate)

#TO DO update_weights

def get_mini_batches(training_data, mini_batch_size):
    return [ training_data[k:k + mini_batch_size] for k in
             xrange(0, len(training_data), mini_batch_size) ]
