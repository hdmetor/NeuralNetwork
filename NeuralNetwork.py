import numpy as np

def sgm(x, der = False):
    #x could be a matrix (and in general will be)
    if not der:
        return 1 / (1 + np.exp(-x))
    else:
        simple = 1 / (1 + np.exp(-x))
        return simple(1 - simple)

class NeuralNetwork:

    def __init__(self, shape, activation = sgm):
        """Create  a new Neural Network with a given shape"""
        # shape = [input, hidden1, ..., hiddenk, output]
        self.shape = shape
        self.activation = activation
        # init weights
        # weight is an array that contains at position i the weight for level i-i to i
        # we added a column to take into account biases
        self.weight = [np.random.randn(j , i+ 1) for i, j in zip(self.shape[:-1], self.shape[1:])]

    def init(self, input, output):
        """Pass the training data to the function"""
        # we divide each 'column' for its max
        # should we scale the data in another way? (x-std)/mean

        #insert a sanity check to make sure that the input passed matches the expected dimension
        # make sure that if the input is already a np array it will not wrapped again
        self.input = np.array(input) / np.amax(input, axis = 0)
        self.output = np.array(output) / np.amax(output)
        #self.cost = sum((self.feed_forward(input) - output))

    def feed_forward(self, input):
        """Pass the input to the network """

        result = input.T
        for w in self.weight:
        # each position in weight represent a level in the network

            # we add a 1 at the end of the input to take bias into account
            new_input = np.vstack([result, np.ones(result.shape[1])])
            result = self.activation(np.dot(w, new_input))


        # this is now the output
        return result

    def back_propagation(self):
        pass



if __name__ == '__main__':

    X = [[3,5], [5,1], [10,2], [1,2], [3,4],[3,5], [5,1], [10,2], [1,2], [3,4]]
    y = [75, 82, 93, 56, 56,75, 82, 93, 56, 56]

    NN = NeuralNetwork([2,5,1])
    NN.init(X,y)

    print(NN.input)
    print(NN.output)

    print("\nresult is")
    print(NN.feed_forward(np.array(X)))
