import mnist_loader
import network
import training

def run():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.NeuralNetwork([784, 30, 10])
    print "Using Cross Entropy Cost"
    training.stochastic_gradient_descent(net, training_data, 30, 10, 0.5,
        l2_regularisation=5.0, evaluation_data=test_data, monitor_evaluation_accuracy=True)

    print "Using Quadratic Cost"
    training.stochastic_gradient_descent(net, training_data, 30, 10, 3.0, cost=training.QuadraticCost, evaluation_data=test_data, monitor_evaluation_accuracy=True)


if __name__ == '__main__':
    run()
