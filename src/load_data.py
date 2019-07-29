from keras.datasets import mnist
import numpy as np
from keras import backend as K


def load_data():
    # Load the dataset
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = np.expand_dims(X_train, axis=3)

    X_test = (X_test.astype(np.float32) - 127.5)/127.5
    X_test = np.expand_dims(X_test, axis=3)

    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    load_data()
