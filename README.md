# Neural Network

A pure `numpy` implementation of a feed forward neural network in Python via Stochastic Gradient Descent with backpropagation.

This is not meant to be a state of the art implementation (no GPU implementation, no convolutions, no dropout ...), more an academic exercise for me to deeply understand the inner details of neural nets. In this respect, it was a very useful and successful project.

The only optimization used is the vectorization when doing mini batch, which gives faster result with respect to regular iteration on lists, but can make the code less clear to understand. I have added some explanations both in the code and in [this file](InterestingBits.ipynb) to clarify key points.

# How it works

TL;DR version: try the [MNIST demo](demo_mnist.py) which (if needed) will download the MNIST data, train a network on the train data, and print the cost on both train and test data at each iteration.

    python demo_mnist.py

Seriously, try it, it works!

## Create the network

First of all we need to create a `NeuralNetwork` object. It can be initialized in two ways:

- with a list containing the number of neurons in each layer:

        import NeuralNetwork as nn
        shape = [2, 5, 1]
        NN = nn.NeuralNetwork(shape)

This creates a neural network with 2 dimensional input layer, a 5 dimensional hidden layer and a 1 dimensional output layer. There is no limit in the number of layers (except memory of course).


- with a string, path to a previously saved config file.

        import NeuralNetwork as nn
        file_location = 'my_saved_network.json'
        NN = nn.NeuralNetwork(file_location)




## The data

The data must be contained in either a list of lists or a `numpy.array` of dimensions `(k, shape[0])` where `shape[0]` is the dimension of the input layer and `k` is the number of data points. 
Similarly, the targets have dimensions `(k, shape[-1])` where `shape[-1]` is the dimensions of the output layer. Think about that as a list where each new example is appended.

Note that in case the data is given as a list of lists it is internally converted into a `numpy.array` for faster matrix multiplication. Such arrays are then transposed for consistency with the current literature.

## Training the network

To train the network we must call the `train` method on the `NeuralNetwork` object. Is it important to pass the `train_data` and `train_labels` variable.

    # given train_data and train_labels and `NN` as before:
    NN.train(train_data=train_data, train_labels=train_labels)

The network will now start training. Once the network is trained we can see what it predicts on some data via

    NN.predict(data)

Technically this could be done at any moment, but an untrained network would just give a random result.

# Advanced training options

The `train` method contains some advanced options not described  in the base tutorial above. We will give here a brief description, together with their default value.

- `batch_size=100`

    Default size of the minibatch used for SGD. Note that `total_examples % batch_size` data point are discarded (if `batch_size` divides `total_examples` we are not discarding any data).

- `epochs=20`

    Number of desired training epochs.

- `eta=.3`

    Learning rate for the network. Note that is remains constant during the whole training process.

- `classification=True`

    The network is training for a classification task. This means that the labels are passed as list of integers (each integer representing a class) and are vectorized automatically. The accuracy is calculated via the `argmax` which does not take into account the confidence of the network for each prediction.

- `print_cost=False`

    If set to true, it will print the value of the cost function at the end of each epochs. Note that this might slow the training, since it requires a new forward passage of the data through the network.

- `test_data=None, test_labels=None`

    If `print_cost` is True, we can also pass test data and labels to print the accuracy on such dataset. As before, this might cause a slowdown due to an extra forward passage.

# Other methods

The `NeuralNetwork` class contains some other methods, namely:

- `NN.save`

Used to save the network data on disk. It dumps a JSON file (with keys `shape`, `weights` and `biases`), where the `np.array` containing the weight and biases are converted to a regular Python list.

    # assume NN is a NeturalNetwork object
    NN.save("my_net.json")

- `NN.load`   

Load a previously saved network from disk. Note that the JSON file must contain all the keys dumped by the `save` method.

- `NN.predict` 

As mentioned before, used to predict on new data. `NN.predict(new_data)` will return a `np.array` with the predicted result.

