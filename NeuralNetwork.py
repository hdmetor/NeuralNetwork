import numpy as np
import pprint as pp
import math

def sgm(x, der=False):
    """Logistic sigmoid function.
    Use der=True for the derivative."""
    if not der:
        return 1 / (1 + np.exp(-x))
    else:
        simple = 1 / (1 + np.exp(-x))
        return simple * (1 - simple)

class NeuralNetwork:
    """Neural Network class.

    Args:
        shape (list): shape of the network. First element is the input layer, last element
        is the output layer.
        activation (optional): pass the activation function. Defaults to sigmoid. 
        """

    def _init_weights(self, shape):
        self.weights = [np.random.randn(j , i) for i, j in zip(shape[:-1], shape[1:])]

    def _init_biases(self, shape):
        self.biases = [np.random.randn(i, 1) for i in self.shape[1:]]
        
    def _init_output(self):
        self.output = []

    def __init__(self, shape, activation=sgm):
        self.shape = shape
        self.activation = activation
        self._init_weights(self.shape)
        self._init_biases(self.shape)
        self._init_output()

    def feed_forward(self, input):
        """Given the input and, return the predicted value according to the current weights."""

        result = input
        self.output.append(result)

        for w, b  in zip(self.weights, self.biases):
            result = self.activation(np.dot(w, result) + b)
            self.output.append(result)

        # the last level is the output
        return result

    def calculate_deltas(self, input, target):
        """ Given the input and the output (typically from a batch),
        it calculates the corresponding deltas.
        """
        # delta for the back propagation
        self.delta = []

        # calculate delta for the output level
        delta = np.multiply( \
                    self.activation(self.output[-1]) - target, \
                    self.activation(self.output[-1], der=True) \
                    )
        self.delta.append(delta)

        # since it's back propagation we start from the end
        steps = len(self.weights) - 1
        for l in range(steps, 0, -1):
            delta = np.multiply(
                        np.dot(
                            self.weights[l].T, 
                            self.delta[steps-l]
                            ),
                        self.activation(self.output[l], der=True)
                        )
            self.delta.append(delta)

        # delta[i] contains the delta for layer i+1
        self.delta.reverse()

    def update_weights(self, total, eta):
        """Use backpropagation to update weights"""
        self.weights =  [self.weights[i] - (eta/total) * np.dot(self.delta[i], self.output[i].T) for i, e in enumerate(self.delta)]

    def update_biases(self, total, eta):
        """Use backpropagation to update the biases"""
        self.biases =  [self.biases[i] - (eta/total)* self.delta[i].sum(axis=1)[:, None] for i, e  in enumerate(self.biases)]

    def cost(self, predicted, target):
        """Calculate the cost function using the current weights and biases"""
        return np.linalg.norm(predicted - target) ** 2

    def SGD(self, input, target, batch_size, epochs = 20, eta = .3, print_cost=False):
        # maybe remove this in the future?
        if isinstance(input, list):
            input = np.array(input)
        if isinstance(target, list):
            target = np.array(target)

        self.input = input.T
        self.target = target.T

        # sanity / shape checks that input / output respect the desired dimensions
        if self.input.shape[0] != self.shape[0]:
            print('Input and shape of the network not compatible')
            exit()
        if self.target.shape[0] != self.shape[-1]:
            print('Output and shape of the network not compatible')
            exit()

        # normalize inputs?
        #self.input = (np.array(input) / np.amax(input, axis = 0)).T
        #self.target = (np.array(target) / np.amax(target)).T
        
        # number of total examples
        total = self.input.shape[1]
        diff = total % batch_size
        # we discard the last examples for now
        if diff != 0:
            self.input = self.input[: total - diff]
            self.target = self.target[: total - diff]
            total = self.input.shape[1]
        
        for epoch in range(epochs):
            # for each epoch, we reshuffle the data and train the network
            print("***** Epoch:", epoch)
            # TODO = each time shuffle the data
            #p = np.random.permutation(len(input))
            #self.input = self.input[p]
            #self.target = self.target[p]
            # create a list of batches (input and target)
            batches_input = [self.input[:, k:k + batch_size] for k in range(0, total, batch_size)]
            batches_target = [self.target[:, k:k + batch_size] for k in range(0, total, batch_size)]
            for batch_input, batch_target in zip(batches_input, batches_target):
                
                # reset the status of the internal variables each time
                self._init_output()
                
                # output values corresponding to the inputs
                self.feed_forward(batch_input)
                
                # do backpropagation
                # calculate delta for all levels
                self.calculate_deltas(batch_input, batch_target)
                # update internal variables
                self.update_weights(batch_size, eta)
                self.update_biases(batch_size, eta)
            
            if (print_cost):
                print("Error is ",
                    self.cost(self.feed_forward(self.input), self.target)
                    )


    def predict(self, data):
        if isinstance(data, list):
            data = np.array(data).T
        return self.feed_forward(data)


if __name__ == '__main__':

    X = []
    y = []
    X_test = []
    y_test = []
    m = 10000

    # sin
    # for i in range(10):
    #     d = (2 * math.pi)/10
    #     X.append([i*d])
    #     y.append([math.sin(i*d)])

    # x^2
    
    # for i in range(m):
    #     X.append([i/m])
    #     y.append([(i/m) ** 2])
    #     X_test.append([(i+1)/m])
    #     y_test.append([((i+1)/m) ** 2])

    # for xx in np.linspace(-1, 1, 1000):
    #     for yy in np.linspace(-1, 1, 1000):
            
    #         X.append([xx, yy])
    #         y.append([np.sin(xx - yy) + np.cos(xx + yy)])

    for xx in np.linspace(-10, 10, 10000):
        X.append([xx])
        y.append([xx ** 2])
    #NN = NeuralNetwork([4, 6,7,10,3, 2])
    NN = NeuralNetwork([1, 2, 1])


    print("starting sgd")
    NN.SGD(X,y,100, epochs=1, print_cost=True)
    print(NN.predict([[0], [1], [-11]]))