import numpy as np
import pprint as pp

def sgm(x, der = False):
    """logistic sigmoid function"""
    if not der:
        return 1 / (1 + np.exp(-x))
    else:
        simple = 1 / (1 + np.exp(-x))
        return simple * (1 - simple)

class NeuralNetwork:

    def init_weights(self, shape):
        self.weights = [np.random.randn(j , i) for i, j in zip(shape[:-1], shape[1:])]

    def init_biases(self, shape):
        self.biases = [np.random.randn( i, 1 ) for i in self.shape[1:]]
        
    def init_output(self):
        self.output = []

    def __init__(self, shape, activation = sgm):
        """Create  a new Neural Network with a given shape
        shape = [input, hidden1, ..., hiddenk, output]"""
        self.shape = shape
        self.activation = activation
        # init weights
        # weight is an array that contains at position i the weight for level i-i to i
        self.init_weights(self.shape)
        self.init_biases(self.shape)
        self.init_output()

    # TODO = remove weights
    def feed_forward(self, input, weights = None):
        """Takes the input and calcualtes the predicted value """
        if weights == None:
            weights = self.weights
        #TODO = remove result, and make this a void function?
        result = input.T
        self.output.append(result)
        #print("input" , input)
        #print("weight", weights)

        for index, w  in enumerate(weights):
        # each position in weight list represent a level in the network

            # calculates the output of the nodes
            print("\t inside feed forwkd, level ", index)
            print("w shape: ", w.shape)
            print("biases shape: ", self.biases[index].shape)
            print("resutl shape:", result.shape)
            
            level_output = np.dot(w, result) + self.biases[index]
            self.output.append(level_output)
            result = self.activation(level_output)

        # the last level is the output
        return result

    def calculate_deltas(self, input, target):
        """ Given the input and the output (typically from a batch),
        it calculates the corresponding deltas,
        """
        # TODO = should delta be a variable in the batch cycle?
        # delta for the back propagation
        self.delta = []

        # calcualte delta for the output level
        delta = np.multiply( \
                    self.activation(self.output[-1]) - target, \
                    self.activation(self.output[-1], der = True) \
                    )
        self.delta.append(delta)

        # since it's back propagation we start from the end
        steps = len(self.weights) - 1
        for l in range(steps, 0, -1):
            delta = np.multiply(
                        np.dot(self.weights[l].T, self.delta[steps-l]),
                        self.activation(self.output[l], der = True)
                        )
            self.delta.append(delta)

        # delta[i] contains the delta for layer i+1
        self.delta.reverse()

        #print("W",[w.shape for w in self.weights])
        #print("B",[w.shape for w in self.biases])
        #print("D",[w.shape for w in self.delta])
        #print("O",[w.shape for w in self.output])

    def gradient_descend(self, eta):
        total = len(self.input) 
        self.update_weights(total, eta)
        self.update_biases(total, eta)
        
    def update_weights(self, total, eta):
        """Use backpropagation to update weights"""
        for (i, d) in enumerate(self.delta):
            print("IN UW it ", i)
            print("weights shape: ", self.weights[i].shape)
            print("delta shape: ",d.shape)
            print("output shape.t: ",self.output[i].T.shape)

            self.weights[i] -= (eta / total) * np.dot(d, self.output[i].T)
        #self.weights =  [self.weights[i] - (eta/total) * np.dot(self.delta[i], self.output[i].T) for i in range(len(self.delta))]

    def update_biases(self, total, eta):
        """Use backpropagation to update the biases"""
        self.biases =  [self.biases[i] - (eta/total)* self.delta[i].sum(axis=1)[:,None] for i in range(len(self.biases))]

    def cost(self):
        """Calculate the cost function using the current weights and output"""
        return np.linalg.norm(self.activation(self.output[-1]) - self.target) ** 2

    def SGD(self, input, target, batch_size, epochs = 20, eta = .3):

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

        total = len(self.input)
        diff = total % batch_size
        # we discard the last examples for now
        if diff != 0:
            self.input = self.input[: total - diff]
            self.target = self.target[: total - diff]
            total = len(self.input)

        if epochs > total:
            # this is only for debug mode
            epochs = total
        for epoch in range(epochs):
            print("Beginning of epoch:", epoch)
            # TODO = each time shuffle the data
            #p = np.random.permutation(len(input))
            #self.input = self.input[p]
            #self.target = self.target[p]
            # create a list of batches (input and target)
            batchesInput = [self.input[:, k:k + batch_size] for k in range(0, total, batch_size)]
            batchesTarget = [self.target[:, k:k + batch_size] for k in range(0, total, batch_size)]
            print(self.input)
            for batchInput, batchTarget in zip(batchesInput, batchesTarget):
                #TODO = possibly init here self.output and the beginning of each iteration 
                # and pass it around as a varible
                print('batch input:')
                pp.pprint(batchInput)
                print('batch target:')
                pp.pprint(batchTarget)
                # pass the input trought the newtork
                print("feeding forward")
                self.feed_forward(batchInput)
                # calcualte delta for all levels
                print("calculating deltas")
                self.calculate_deltas(batchInput, batchTarget)
                # updating weights and biases
                print("updating weights")
                self.update_weights(batch_size, eta)
                self.update_biases(batch_size, eta)


if __name__ == '__main__':

    X = [[3,5], [5,1], [10,2], [1,2], [3,4],[3,5], [5,1], [10,2], [1,2], [3,4]]
    y = [75, 82, 93, 56, 56,75, 82, 93, 56, 56]


    X, y  = [[1,2]], [3]

    X, y = [[1,2,3,4],[11,22,33,44],[21,22,32,24],[2,3,4,5],[52,53,54,55]], [[10,11],[210,211],[310,311], [422,423], [523,525]]


    #NN = NeuralNetwork([4, 6,7,10,3, 2])
    NN = NeuralNetwork([4, 6, 2])
    """
    NN.init(X,y)


    NN.feed_forward()
    NN.back_propagation()
    NN.gradient_descend(eta = .3)
"""
    print("starting sgd")
    NN.SGD(X,y,2)
    #c1 = NN.cost()
    #NN.feed_forward()
    #c2 = NN.cost()
    #print('is cost decreasing? ',c2 < c1)

"""
TODOS:

- insert a save and load method

"""
