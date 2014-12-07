import numpy as np

def sgm(x, der = False):
    #x could be a matrix (and in general will be)
    if not der:
        return 1 / (1 + np.exp(-x))
    else:
        simple = 1 / (1 + np.exp(-x))
        return simple * (1 - simple)

class NeuralNetwork:

    def __init__(self, shape, activation = sgm):
        """Create  a new Neural Network with a given shape"""
        # shape = [input, hidden1, ..., hiddenk, output]
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

        #insert a sanity check to make sure that the input (and target) passed matches the expected dimensions
        # make sure that if the input is already a np array it will not wrapped again
        self.input = np.array(input) / np.amax(input, axis = 0)
        self.target = np.array(target) / np.amax(target)
        #self.cost = sum((self.feed_forward(input) - target))

    def feed_forward(self, input, weights = None):
        """Pass the input to the network """
        if weights == None:
            weights = self.weights

        # is this correct?
        self.output.append(input)
        result = input.T


        for index, w  in enumerate(weights):
        # each position in weight represent a level in the network


            # this is not taking into account the biases
            #this contains all the z^l = sum(w*a^(l-1)+b)
            level_output = np.dot(w, result) + self.biases[index]
            self.output.append(level_output)
            result = self.activation(level_output)


        # this is now the target
        return result

    def back_propagation(self):
        # this will contain the delta for the back propagation
        self.delta = []

        # delta for the output level
        delta = np.multiply( \
                    self.activation(self.output[-1]) - self.target, \
                    self.activation(self.output[-1], der = True) \
                    )
        self.delta.append(delta)

        steps = len(self.weights) - 1
        for l in range(steps, 0, -1):
            delta = np.multiply(
                        np.dot(self.weights[l].T,self.delta[steps-l]),
                        self.activation(self.output[l-1], der = True)
                        )
            self.delta.append(delta)

        # delta[i] contains the delta for level i
        self.delta.reverse()
        

    def update_weights(self,  eta = 0.3):
        total = 1
        return [self.weights[i] - (eta/total) * np.dot(self.delta[i], self.output[i].T) for i in range(len(self.delta))]


    def update_biases(self, eta = 0.3):
        # this is going to be the number of the training examples
        total = 1
        return [self.biases[i] - (eta/total)* self.delta[i] for i in range(len(self.biases))] 
        



if __name__ == '__main__':

    X = [[3,5], [5,1], [10,2], [1,2], [3,4],[3,5], [5,1], [10,2], [1,2], [3,4]]
    y = [75, 82, 93, 56, 56,75, 82, 93, 56, 56]


    X, y  = [[1,2]], [3]



    NN = NeuralNetwork([2, 3, 1])
    NN.init(X,y)

    #print(NN.input)
    #print(NN.target)

    #print("\nresult is")
    #print(NN.feed_forward(np.array(X)))

    NN.feed_forward(np.array(X))
    NN.back_propagation()
    NN.update_biases()
    NN.update_weights()
