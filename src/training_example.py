import mnist_loader
import network
import training

def run():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.NeuralNetwork([784, 30, 10])
    training.stochastic_gradient_descent(net, training_data, 30, 10, 3.0, test_data=test_data)


if __name__ == '__main__':
    run()
