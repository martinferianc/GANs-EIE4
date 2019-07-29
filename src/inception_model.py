"""
Code adapted from: https://github.com/eriklindernoren/Keras-GAN
"""
from gan import GAN
from keras import backend as K

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import load_model



import numpy as np

class LeNet(GAN):
    def __init__(self, lr = 0.001, name = "LeNet"):
        super(LeNet, self).__init__(name)

        # Build and compile the basic lenet model
        self.model = self.build_model()
        self.model.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr),
            metrics=['accuracy'])

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(28,28,1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.class_dim, activation='softmax'))

        model.summary()

        return model

    def load_model(self, model_path):
        self.model = load_model(model_path+".h5")

    def train(self, X_train, y_train, epochs, batch_size=128, save_interval=50, save=False, X_test=None, y_test=None):

        self.model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(X_test, y_test))

        if save:
            self.model.save("models/"+self.name+".h5")
    def predict(self, X):
        return self.model.predict(X)
    def evaluate(self,X, y):
        if self.model is None:
            raise ValueError('Load or train a model!')

        return self.model.evaluate(X, y, verbose=0)
