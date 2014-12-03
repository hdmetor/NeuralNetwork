import numpy as np

class NeuralNetwork:

    def __init__(self, shape):
        # shape = [input, hidden, ouput]
        self.shape = shape

        # init weights
        self.weights1 = np.random.rand(self.shape[0], self.shape[1])
        self.weights2 = np.random.rand(self.shape[1], self.shape[2])

    def init(self, input, output):
        # we divide each 'column' for its max
        # should we scale the data in another way? (x-std)/mean
        self.input = np.array(input) / np.amax(input, axis = 0)
        self.output = np.array(output) / np.amax(output)



    def feed_forward(self):
        X = self.input
        self.z2 =  np.dot(X, self.weights1)
        self.a2 = sgm(self.z2)
        self.z3 = np.dot(self.a2, self.weights2)

        return sgm(self.z3)

def sgm(x, der = False):
    #x could be a matrix (and in general will be)
    if not der:
        return 1 / (1 + np.exp(-x))
    else:
        simple = 1 / (1 + np.exp(-x))
        return simple(1 - simple)

if __name__ == '__main__':

    X = [[3,5],[5,1],[10,2]]
    y = [75,82, 93]
    NN = NeuralNetwork(X,y)

    print(NN.input)
    print(NN.output)

    print("\nresult is")
    print(NN.feed_forward())
