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

        delta = []

        # delta for the output level
        delta_l = (self.activation(self.output[-1]) - self.target) * self.activation(self.output[-1], der = True)
        delta.append(delta_l)

        steps = len(self.weights) - 1
        for l in range(steps, -1, -1):
            print('step is', l, steps-l)
            print('shape ',self.weights[l].T.shape)
            delta_l = np.dot(self.weights[l].T,delta[steps-l])
            delta.append(delta_l)
            pass


        print(len(self.weights),"delta\n\n",delta)
        pass


if __name__ == '__main__':

    X = [[3,5], [5,1], [10,2], [1,2], [3,4],[3,5], [5,1], [10,2], [1,2], [3,4]]
    y = [75, 82, 93, 56, 56,75, 82, 93, 56, 56]


    #X, y  = [[1,2]], [3]



    NN = NeuralNetwork([2,5,3,1])
    NN.init(X,y)

    #print(NN.input)
    #print(NN.target)

    #print("\nresult is")
    #print(NN.feed_forward(np.array(X)))


    NN.back_propagation()
