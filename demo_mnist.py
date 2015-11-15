from __future__ import print_function
from __future__ import division

import data_load as dl
import NeuralNetwork as nn


def main():
    shape = [784, 50, 30, 10]
    net = nn.NeuralNetwork(shape, activation=nn.sgm)

    print("Gathering the training data")
    X_train, y_train = dl.get_images_and_labels('train')
    assert (X_train.shape, y_train.shape) == ((60000, 28, 28),
                                              (60000, 1)), "Train images were loaded incorrectly"
    X_train = X_train.reshape(60000, 784)

    print("Gathering the test data")
    X_test, y_test = dl.get_images_and_labels('test')
    assert (X_test.shape, y_test.shape) == ((10000, 28, 28),
                                            (10000, 1)), "Test images were loaded incorrectly"
    X_test = X_test.reshape(10000, 784)

    print("Starting the training")
    net.train(
            train_data=X_train,
            train_labels=y_train,
            batch_size=200,
            epochs=200,
            learning_rate=3.,
            print_cost=True,
            test_data=X_test,
            test_labels=y_test,
            plot=True)


if __name__ == '__main__':
    main()
