import numpy as np
import json

def sgm(x, der=False):
    """Logistic sigmoid function.
    Use der=True for the derivative."""
    if not der:
        return 1 / (1 + np.exp(-x))
    else:
        simple = 1 / (1 + np.exp(-x))
        return simple * (1 - simple)

def relu(x, der=False):
    """Rectifier activation function.
    Use der=True for the derivative."""
    if not der:
        return np.maximum(0, x)
    else:
        if x <= 0:
            return 0
        else:
            return 1


class NeuralNetwork:
    """Neural Network class.

    Args:
        shape (list): shape of the network. First element is the input layer, last element
        is the output layer.
        activation (optional): pass the activation function. Defaults to sigmoid. 
        """

    # outputs: output of the layers (before the sigmoid)
    # activations: outputs after the sigmoid
    def _init_weights(self):
        self.weights = [np.random.randn(j , i) for i, j in zip(self.shape[:-1], self.shape[1:])]
    def _init_biases(self):
        self.biases = [np.random.randn(i, 1) for i in self.shape[1:]]
        
    def _init_activations(self):
        self.activations = []

    def _init_outputs(self):
        self.outputs = []

    def _init_deltas(self):
        self.deltas = []

    def __init__(self, shape=[], activation=sgm, file=False):
        if file:
            self.load(file)
        else:
            self.shape = shape
            self.activation = activation
            self._init_weights()
            self._init_biases()
            self._init_activations()
            self._init_outputs()

    def vectorize_output(self):
        """Tranforms a categorical label represented by an integer into a vector."""
        num_labels = np.unique(self.target).shape[0]
        num_examples = self.target.shape[1]
        result = np.zeros((num_labels, num_examples))
        for l, c in zip(self.target.ravel(), result.T):
            c[l] = 1
        self.target = result

    def labelize(self, data):
        """Tranform a matrix (where each column is a data) into an list that contains the argmax of each item."""

        return np.argmax(data, axis=0)


    def feed_forward(self, data, return_labels=False):
        """Given the input and, return the predicted value according to the current weights."""
        result = data
        # if z = w*a +b
        # then activations are sigma(z)

        self._init_outputs()
        self._init_activations()

        self.activations.append(data)
        self.outputs.append(data)

        for w, b  in zip(self.weights, self.biases):
            result = np.dot(w, result) + b
            self.outputs.append(result)
            result = self.activation(result)
            self.activations.append(result)

        if return_labels:
            result = self.labelize(result)
                
        # the last level is the activated output
        return result

    def calculate_deltas(self, data, target):
        """ Given the input and the output (typically from a batch),
        it calculates the corresponding deltas.
        It is assumed that the network has just feed forwarded its batch.
        Deltas are stored in a (n, k) matrix, where n is the dimensions of the 
        corresponding layer and k is the number of examples.
        """
        # delta for the back propagation
        self._init_deltas()

        # calculate delta for the output level
        delta = np.multiply( \
                    self.activations[-1] - target, \
                    self.activation(self.outputs[-1], der=True) \
                    )
        self.deltas.append(delta)

        # since it's back propagation we start from the end
        steps = len(self.weights) - 1
        for l in range(steps, 0, -1):
            delta = np.multiply(
                        np.dot(
                            self.weights[l].T, 
                            self.deltas[steps-l]
                            ),
                        self.activation(self.outputs[l], der=True)
                        )
            self.deltas.append(delta)

        # delta[i] contains the delta for layer i+1
        self.deltas.reverse()

    def update_weights(self, total, eta):
        """Use backpropagation to update weights"""
        self.weights =  [w - (eta/total) * np.dot(d, a.T) for w, d, a in zip(self.weights, self.deltas, self.activations)]

    def update_biases(self, total, eta):
        """Use backpropagation to update the biases"""
        # summing over the columns of d, as each column is a different example
        self.biases = [b - (eta/total)* (np.sum(d, axis=1)).reshape(b.shape) for b, d  in zip(self.biases, self.deltas)]

    def cost(self, predicted, target):
        """Calculate the cost function using the current weights and biases"""
        # the cost is normalized (divided by numer of samples)
        if self.classification:
            return np.sum(predicted != target) / len(predicted)
        else:
            return (np.linalg.norm(predicted - target) ** 2)  / predicted.shape[1]


    def train(self, data, target, batch_size, epochs=20, eta=.3, print_cost=False, classification=True, method='SGD'):
        """Train the network using the """
        if method is not 'SGD':
            print ("This method is not supported at the moment")
            exit()

        self.classification = classification
        # maybe remove this in the future?
        if isinstance(data, list):
            data = np.array(data)
        if isinstance(target, list):
            target = np.array(target)

        self.data = data.T
        self.target = target.T
        if self.classification:
            self.original_labels = self.target.ravel()
            self.vectorize_output()

        # sanity / shape checks that input / output respect the desired dimensions
        if self.data.shape[0] != self.shape[0]:
            print('Input and shape of the network not compatible: '.
                self.data.shape[0], " != ", self.shape[0])
            exit()
        if self.target.shape[0] != self.shape[-1]:
            print('Output and shape of the network not compatible: ', 
                self.target.shape[0], " != ", self.shape[-1])
            exit()

        # normalize inputs?
        #self.input = (np.array(input) / np.amax(input, axis = 0)).T
        #self.target = (np.array(target) / np.amax(target)).T
        
        # number of total examples
        self.number_of_examples = self.data.shape[1]
        diff = self.number_of_examples % batch_size
        # we discard the last examples for now
        if diff != 0:
            self.data = self.data[: self.number_of_examples - diff]
            self.target = self.target[: self.number_of_examples - diff]
            self.number_of_examples = self.data.shape[1]
        
        for epoch in range(epochs):
            # for each epoch, we reshuffle the data and train the network
            print("Starting epoch:", epoch,"/",epochs - 1, end="")
            # create a list of batches (input and target)
            permutation = np.random.permutation(self.number_of_examples)
            # we transpose twice to permutate over the columns
            self.data = self.data.T[permutation].T
            self.target = self.target.T[permutation].T
            self.original_labels = self.original_labels[permutation]
            batches_input = [self.data[:, k:k + batch_size] for k in range(0, self.number_of_examples, batch_size)]
            batches_target = [self.target[:, k:k + batch_size] for k in range(0, self.number_of_examples, batch_size)]
            for batch_input, batch_target in zip(batches_input, batches_target):
                # reset the status of the internal variables each time
                self._init_outputs()
                self._init_activations()

                # feed forward the input
                self.feed_forward(batch_input)

                # do backpropagation: calculate deltas for all levels
                self.calculate_deltas(batch_input, batch_target)
                
                # update internal variables
                self.update_weights(batch_size, eta)
                self.update_biases(batch_size, eta)

            if print_cost:
                if self.classification:
                    cost = self.cost(
                            self.feed_forward(self.data, return_labels=True), 
                            self.original_labels
                            )
                    print("\terror is {0:.2f}%\n".format(cost * 100
                        ))
                    pass
                else:
                    forwarded = self.feed_forward(self.data)
                    print("\terror is \n", self.cost(forwarded, self.target))


    def predict(self, data):
        if isinstance(data, list):
            data = np.array(data).T
        return self.feed_forward(data)

    def save(self, file_location):
        """Save network's data in a JSON file located in file_location"""
        data = {
            "shape" : self.shape,
            "weights" : [w.tolist() for w in self.weights],
            "biases" : [b.tolist() for b in self.biases]
        }
        with open(file_location, 'w') as fp:
            json.dump(data, fp)

    def load(self, file_location):
        with open(file_location, 'r') as fp:
            data = json.load(fp)
        self.shape = data["shape"]
        self.weights = [np.array(w) for w in data["weights"]]
        self.biases = [np.array(b) for b in data["biases"]]


  