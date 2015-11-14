"""MNist data loader. Some of this code comes from
http://martin-thoma.com/classify-mnist-with-pybrain/
"""

from __future__ import print_function
from __future__ import division

from struct import unpack
import gzip
import numpy as np
import os
from urllib.request import urlretrieve
from functools import partial
from collections import deque
import requests

LECUN = 'http://yann.lecun.com/exdb/mnist/'

TRAIN = { "data" : 'train-images-idx3-ubyte.gz', "labels" : 'train-labels-idx1-ubyte.gz'}
TEST = { "data" :'t10k-images-idx3-ubyte.gz', "labels" : 't10k-labels-idx1-ubyte.gz'}

FOLDER = 'data'

def get_images_and_labels(train_or_test, folder=FOLDER):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """

    if not os.path.exists(folder):
        os.makedirs(folder)

    if train_or_test == 'train':
        files = TRAIN
    elif train_or_test == 'test':
        files = TEST
    else:
        print("The second argument must be 'train' or 'test'")
        return

    # checks if each file is present on disk
    # if not, it downloads it
    deque(
        map(
            partial(check_or_download, data_folder=folder), 
            files.values()
        )
    )
    return read_images(files["data"], folder), read_labels(files["labels"], folder)

def read_images(file_name, data_folder):
    file_location = os.path.join(data_folder, file_name)
    with gzip.open(file_location, 'rb') as images:
        images.read(4)
        number_of_images = images.read(4)
        number_of_images = unpack('>I', number_of_images)[0]
        rows = images.read(4)
        rows = unpack('>I', rows)[0]
        cols = images.read(4)
        cols = unpack('>I', cols)[0]
        x = np.zeros((number_of_images, rows, cols), dtype=np.uint8) 

        for i in range(number_of_images):
            if i % int(number_of_images / 10) == int(number_of_images / 10) - 1:
                print("Reading images progress ", int(100 * (i + 1) / number_of_images) , "%")
            for row in range(rows):
                for col in range(cols):
                    tmp_pixel = images.read(1)  # Just a single byte
                    tmp_pixel = unpack('>B', tmp_pixel)[0]
                    x[i][row][col] = tmp_pixel

    return x

def read_labels(file_name, data_folder):
    file_location = os.path.join(data_folder, file_name)
    with gzip.open(file_location, 'rb') as labels:
        labels.read(4)
        number_of_labels = labels.read(4)
        number_of_labels = unpack('>I', number_of_labels)[0]
        y = np.zeros((number_of_labels, 1), dtype=np.uint8) 

        for i in range(number_of_labels):
            tmp_label = labels.read(1)
            y[i] = unpack('>B', tmp_label)[0]
    
    return y


def check_or_download(file_name, data_folder, url=LECUN):
    file_location = os.path.join(data_folder, file_name)
    if not os.path.exists(file_location):
        print("Downloading ", file_name)
        page = requests.get(url + file_name)
        with open(file_location, 'wb') as fp:
            fp.write(page.content)




