# Neural Network

A pure `numpy` implementation of a feed forward neural network in Python via Stochastic Gradient Descent with backpropagation.

This is not meant to be a state of the art implementation (no GPU implementation, no convolutions, no dropout ...), it was more an academic exercise for me to deeply understand the inner details of neural nets. In this respect, it was a very useful and successful project.

The only optimization used is the vectorization when doing mini batch, which gives faster result with respect to regular iteration on lists, but can make the code less clear to understand. I have added some explanations both in the code and in [this file](InterestingBits.ipynb) to clarify key points.

# How it works

## Create the network

First of all we need to create a `NeuralNetwork` object. It can be initialized in two ways:

- with a list containing the number of neurons in each layer:

        import NeuralNetwork as nn
        shape = [2, 5, 1]
        NN = nn.NeuralNetwork(shape)

This creates a neural network with 2 dimensional input layer, a 5 dimensional hidden layer and a 1 dimensional output layer. There is no limit in the number of layers (except memory of course).


- with a string, path to a previously saved file.

        import NeuralNetwork as nn
        file_location = 'my_saved_network.json'
        NN = nn.NeuralNetwork(file_location)




## The data

The data must be contained in either a list of lists or a `numpy.array` of dimensions `(k, shape[0])` where `shape[0]` is the dimension of the input layer and `k` is the number of data points. 
Similarly, the targets have dimensions `(k, shape[-1])` where `shape[-1]` is the dimensions of the output layer. Think about that as a list that is incremented each time with a new example.

Note that in case the data is given as a list of lists it is internally converted into a `numpy.array` for faster matrix multiplication. Such arrays are then transposed for consistency with the current literature.

## Training the network

To train the network we must call the `train` method on the `NeuralNetwork` object. Is it important to pass the `train_data` and `train_labels` variable.

    # given train_data and train_labels and `NN` as before:
    NN.train(train_data=train_data, train_labels=train_labels)

The network will now start training. Once the network is trained we can predict the result of some example via

    NN.predict(new_data)

Technically this could be done at any moment, but an untrained network would just give a random result.



