import numpy as np

def _sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def _sigmoid_prime(z):
    return _sigmoid(z)*(1 - _sigmoid(z))

sigmoid = np.vectorize(_sigmoid)
sigmoid_prime = np.vectorize(_sigmoid_prime)
