import mnist_loader
import network
import training

def run():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.NeuralNetwork([784, 100, 10])
    print "Using Cross Entropy Cost"
    training.stochastic_gradient_descent(net, training_data, 60, 10, 0.1,
        l2_regularisation=5.0, evaluation_data=validation_data, monitor_evaluation_accuracy=True)

    net.save("network.json")
    print "saved as network.json"

if __name__ == '__main__':
    run()
