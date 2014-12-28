import numpy as np

def sgm(x, der = False):
    """logistic sigmoid function"""
    if not der:
        return 1 / (1 + np.exp(-x))
    else:
        simple = 1 / (1 + np.exp(-x))
        return simple * (1 - simple)

class NeuralNetwork:

    def __init__(self, shape, activation = sgm):
        """Create  a new Neural Network with a given shape
        shape = [input, hidden1, ..., hiddenk, output]"""
        self.shape = shape
        self.activation = activation
        # init weights
        # weight is an array that contains at position i the weight for level i-i to i
        # we added a column to take into account biases
        self.weights = [np.random.randn(j , i) for i, j in zip(self.shape[:-1], self.shape[1:])]
        self.biases = [np.random.randn( i, 1 ) for i in self.shape[1:]]
        self.output = []

    def init(self, input, target):
        """Pass the training data to the function"""
        # we divide each 'column' for its max
        # should we scale the data in another way? (x-std)/mean

        # make sure that if the input is already a np array it will not wrapped again
        self.input = (np.array(input) / np.amax(input, axis = 0)).T
        self.target = (np.array(target) / np.amax(target)).T

        if self.input.shape[0] != self.shape[0]:
            print('Input and shape of the network not compatible')
            exit()

        if self.target.shape[0] != self.shape[-1]:
            print('Output and shape of the network not compatible')
            exit()

    def feed_forward(self, input = None, weights = None):
        """Pass the input to the network """
        if weights == None:
            weights = self.weights

        if input == None:
            input = self.input

        result = input
        self.output.append(result)



        for index, w  in enumerate(weights):
        # each position in weight list represent a level in the network

            # caluclates the output of the nodes
            level_output = np.dot(w, result) + self.biases[index]
            self.output.append(level_output)
            result = self.activation(level_output)

        # the last level is the output
        return result

    def back_propagation(self):
        # delta for the back propagation
        self.delta = []

        # calucalte delta for the output level
        delta = np.multiply( \
                    self.activation(self.output[-1]) - self.target, \
                    self.activation(self.output[-1], der = True) \
                    )
        self.delta.append(delta)

        # since it's back propagation we start from the end
        steps = len(self.weights) - 1
        for l in range(steps, 0, -1):
            delta = np.multiply(
                        np.dot(self.weights[l].T,self.delta[steps-l]),
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
        self.weights =  [self.weights[i] - (eta/total) * np.dot(self.delta[i], self.output[i].T) for i in range(len(self.delta))]


    def update_biases(self, total, eta):
        """Use backpropagation to update the biases"""
        self.biases =  [self.biases[i] - (eta/total)* self.delta[i].sum(axis=1)[:,None] for i in range(len(self.biases))]


    def cost(self):
        """Calculate the cost function using the current weights and output"""
        return np.linalg.norm(self.activation(self.output[-1]) - self.target) ** 2

if __name__ == '__main__':

    X = [[3,5], [5,1], [10,2], [1,2], [3,4],[3,5], [5,1], [10,2], [1,2], [3,4]]
    y = [75, 82, 93, 56, 56,75, 82, 93, 56, 56]


    X, y  = [[1,2]], [3]

    X, y = [[1,2,3,4],[1,2,3,4],[1,2,3,4],[2,3,4,5],[2,3,4,5]], [[10,11],[10,11],[10,11], [22,23], [23,25]]


    #NN = NeuralNetwork([4, 6,7,10,3, 2])
    NN = NeuralNetwork([4, 6, 2])
    NN.init(X,y)


    NN.feed_forward()
    NN.back_propagation()
    NN.gradient_descend(eta = .3)

    c1 = NN.cost()
    NN.feed_forward()
    c2 = NN.cost()
    print('is cost decreasing? ',c2 < c1)

 
