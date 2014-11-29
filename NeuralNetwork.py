import numpy as np

class NeuralNetwork:

    def __init__(self, input, output):
        self.shape = [2,3,1]
        # we divide each 'column' for its max
        # should we scale the data in another way? (x-std)/mean
        self.input = np.array(input) / np.amax(input, axis = 0)
        self.output = np.array(output) / np.amax(output)


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
