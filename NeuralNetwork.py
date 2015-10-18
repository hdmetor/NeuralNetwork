import numpy as np
import pprint as pp
import math

def sgm(x, der=False):
    """logistic sigmoid function"""
    if not der:
        return 1 / (1 + np.exp(-x))
    else:
        simple = 1 / (1 + np.exp(-x))
        return simple * (1 - simple)

class NeuralNetwork:

    def _init_weights(self, shape):
        self.weights = [np.random.randn(j , i) for i, j in zip(shape[:-1], shape[1:])]

    def _init_biases(self, shape):
        self.biases = [np.random.randn( i, 1 ) for i in self.shape[1:]]
        
    def _init_output(self):
        self.output = []

    def __init__(self, shape, activation=sgm):
        """Create  a new Neural Network with a given shape
        shape = [input, hidden_1, ..., hidden_k, output]
        using the function activation"""
        self.shape = shape
        self.activation = activation
        # init weights
        # weight is an array that contains at position i the weight for level i-i to i
        self._init_weights(self.shape)
        self._init_biases(self.shape)
        self._init_output()
        print (" weight init ")
        for w in self.weights:
            print(w.shape)

    # TODO = remove weights
    def feed_forward(self, input):
        """Given the input and, return the predicted value according to the current weights."""

        result = input
        self.output.append(result)
        #print("input" , input)
        #print("weight", weights)

        # TODO = use zip
        for w, b  in zip(self.weights, self.biases):
        # each position in weight list represent a level in the network

            # calculates the output of the nodes
            # print("\t inside feed forward, level ", index)
            # print("w shape: ", w.shape)
            # print("biases shape: ", self.biases[index].shape)
            # print("result shape:", result.shape)

            result = self.activation(np.dot(w, result) + b)
            self.output.append(result)
            # result = self.activation(level_output)

        # the last level is the output
        return result

    def calculate_deltas(self, input, target):
        """ Given the input and the output (typically from a batch),
        it calculates the corresponding deltas.
        """
        # TODO = should delta be a variable in the batch cycle?
        # delta for the back propagation
        self.delta = []

        # calcualte delta for the output level
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

        #print("W",[w.shape for w in self.weights])
        #print("B",[w.shape for w in self.biases])
        #print("D",[w.shape for w in self.delta])
        #print("O",[w.shape for w in self.output])


    def update_weights(self, total, eta):
        """Use backpropagation to update weights"""
        # for (i, d) in enumerate(self.delta):
        #     print("IN UW it ", i)
        #     print("weights shape: ", self.weights[i].shape)
        #     print("delta shape: ",d.shape)
        #     print("output shape.t: ",self.output[i].T.shape)

        #     self.weights[i] -= (eta / total) * np.dot(d, self.output[i].T)
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
        print("epochs ", epochs, total, )
        if epochs > total:
            # this is only for debug mode
            epochs = total
        print("epochs ", epochs)
        for epoch in range(epochs):
            # for each epoch, we reshuffle the data and train the network
            print("*****Beginning of epoch:", epoch)
            # TODO = each time shuffle the data
            #p = np.random.permutation(len(input))
            #self.input = self.input[p]
            #self.target = self.target[p]
            # create a list of batches (input and target)
            batchesInput = [self.input[:, k:k + batch_size] for k in range(0, total, batch_size)]
            batchesTarget = [self.target[:, k:k + batch_size] for k in range(0, total, batch_size)]
            for batchInput, batchTarget in zip(batchesInput, batchesTarget):
                
                self._init_output()
                
                # print('batch input:')
                # pp.pprint(batchInput)
                # print('batch target:')
                # pp.pprint(batchTarget)
                # pass the input trought the newtork
                # print("feeding forward")
                self.feed_forward(batchInput)
                # calcualte delta for all levels
                # print("calculating deltas")
                self.calculate_deltas(batchInput, batchTarget)
                # updating weights and biases
                # print("updating weights")
                self.update_weights(batch_size, eta)
                self.update_biases(batch_size, eta)
                if (print_cost):
                    print("Error is ", self.cost(1, 2))


    def predict(self, data):
        result = np.array(data).T
        #output = np.array(output).T
        for w, b in zip(self.weights, self.biases):
        
            result = np.dot(w, result) + b
        return result.T
        #print("cost was ", self.cost(result, output))


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

    for xx in np.linspace(-1, 1, 1000):
        for yy in np.linspace(-1, 1, 1000):
            
            X.append([xx, yy])
            y.append([np.sin(xx - yy) + np.cos(xx + yy)])


    #NN = NeuralNetwork([4, 6,7,10,3, 2])
    NN = NeuralNetwork([2, 4, 1])

    print("starting sgd")
    NN.SGD(X,y,100, epochs=4)
    #NN.test(X_test, y_test)
    print(NN.predict([[0,0]]))



"""
TODOS:

- insert a save and load method

"""
