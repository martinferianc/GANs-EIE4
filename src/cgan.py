from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from gan import GAN
from tqdm import tqdm
from inception_score import *
import json

import numpy as np

class CGAN(GAN):
    """ Conditional GAN, as per https://arxiv.org/abs/1411.1784
    We base our GAN architecture on a DCGAN model
    """

    def __init__(self, name, d ,f):
        super(CGAN, self).__init__(name)
        self.depth = d
        self.filters = f
        self.build_model()

    def build_model(self):
        # Input Tensors
        self.input_G = Input(shape=(self.noise_dim,)) # Noise Vector
        self.input_D = Input(shape=self.img_shape) # Image Tensor
        self.conditioning_label = Input(shape=(self.class_dim,))  # One-hot encoded label

        # Assemble CGAN Model using the functional API
        self.G = self.generator(self.input_G, self.conditioning_label)
        self.D = self.discriminator(self.input_D, self.conditioning_label)
        self.D.trainable = True
        self.D.compile(Adam(self.lr_d, beta_1 = 0.5), "binary_crossentropy", metrics = ["accuracy"])
        self.D.trainable = False
        self.m = Model([self.input_G, self.conditioning_label], self.D([self.output_G, self.conditioning_label]))
        self.m.compile(Adam(self.lr_g, beta_1 = 0.5), "binary_crossentropy", metrics = ["accuracy"])

        plot_model(self.G, to_file="../../Figures/CW2/{}/G_{}.png".format(self.name,self.name))
        plot_model(self.D, to_file="../../Figures/CW2/{}/D_{}.png".format(self.name,self.name))

    def train(self, X_train, y_train, inception_model, epochs=10, sample_interval=1, batch_size=128, save= False):
        """ Train CGAN:
            - Train D to discriminate G results, conditioned on label
            - Train G to fool D, conditioned on label
        """
        self.batch_size = batch_size

        y_g = [1]*self.batch_size
        y_d_true = [1]*self.batch_size
        y_d_gen = [0]*self.batch_size

        for epoch in range(epochs):
            print ("---- Epoch %d ----" % (epoch+1))

            discriminator_loss_epoch = []
            discriminator_accuracy_epoch = []
            generator_loss_epoch = []

            if self.virtual_batch_norm:
                reference_idx = np.random.randint(0, X_train.shape[0], int(self.batch_size//2))
                reference_imgs, reference_y = X_train[reference_idx], y_train[reference_idx]

            for i in tqdm(range(X_train.shape[0] // self.batch_size)):

                # Select a random half batch of images
                X_d_true = None
                y = None
                if self.virtual_batch_norm:
                    idx = np.random.randint(0, X_train.shape[0], self.batch_size//2)
                    imgs_random = X_train[idx]
                    y_random =  y_train[idx]

                    # Add them together
                    imgs = np.concatenate((imgs_random, reference_imgs), axis = 0)
                    y = np.concatenate((y_random, reference_y), axis = 0)

                    mean = np.mean(imgs, axis=0)
                    std = np.std(imgs, axis = 0)
                    std[std==0] = 1
                    imgs = (imgs - mean)/std

                    # Reshuffle the images
                    p = np.random.permutation(len(imgs))
                    X_d_true = imgs[p]
                    y = y[p]

                else:
                    idx = np.random.randint(0, X_train.shape[0], self.batch_size)
                    X_d_true, y = X_train[idx], y_train[idx]

                if self.label_smoothing:
                    for row in y:
                        i = np.argmax(row)
                        row[i] = i / 10

                X_g = np.array([np.random.normal(0,0.5,100) for _ in range(self.batch_size)])
                X_d_gen = self.G.predict([X_g,y], verbose=0)

                # Train discriminator
                d_loss = self.D.train_on_batch([X_d_true, y], y_d_true)
                d_loss += self.D.train_on_batch([X_d_gen, y], y_d_gen)

                prediction_true = self.D.predict_on_batch([X_d_true,y])
                prediction_gen = self.D.predict_on_batch([X_d_gen,y])

                d_accuracy = 0
                for i in range(len(prediction_true)):
                    if int(prediction_true[i]) == y_d_true[i]:
                        d_accuracy+=1

                for i in range(len(prediction_gen)):
                    if int(prediction_gen[i]) == y_d_gen[i]:
                        d_accuracy+=1

                d_accuracy/=(len(prediction_true)+ len(prediction_gen))

                # Train generator i.e. whole model (G + frozen D)
                g_loss = self.m.train_on_batch([X_g, y], y_g)


                discriminator_loss_epoch.append(d_loss)
                discriminator_accuracy_epoch.append(d_accuracy)
                generator_loss_epoch.append(g_loss)

            self.discriminator_loss.append(np.asscalar(np.array(discriminator_loss_epoch).mean()))
            self.discriminator_accuracy.append(np.asscalar(np.array(discriminator_accuracy_epoch).mean()))
            self.generator_loss.append(np.asscalar(np.array(generator_loss_epoch).mean()))


            # Calculate inception score
            noise = np.random.uniform(-1, 1, (100 * self.class_dim, self.noise_dim))
            noise_labels = []
            for i in range(self.class_dim):
                noise_labels+=[i]*100
            noise_labels = to_categorical(noise_labels, self.class_dim)

            generated_images = self.G.predict([noise,noise_labels], verbose=0)
            score = inception_score(inception_model, generated_images, noise_labels)

            self.inception_score.append(score)


            print ("[D loss: %f, acc.: %.2f%%] [G loss: %f] [I score: %f]" % (self.discriminator_loss[epoch],
                                                            self.discriminator_accuracy[epoch],
                                                            self.generator_loss[epoch],
                                                            self.inception_score[epoch]))
            if ((epoch+1)% sample_interval == 0) or (epoch ==0):
                self.sample_images(epoch)

            if save:
                self.G.save("models/"+self.name+"_generator.h5")
                self.D.save("models/"+self.name+"_discriminator.h5")
                self.m.save("models/"+self.name+"_complete.h5")
                history = {"discriminator_loss": self.discriminator_loss,
                           "discriminator_accuracy": self.discriminator_accuracy,
                           "generator_loss": self.generator_loss,
                           "inception_score": self.inception_score}
                with open("models/"+self.name+"_history.json", 'w') as outfile:
                    json.dump(history, outfile)

        return history

    def generator(self, input_G, conditioning_label):
        """ CGAN Generator, small neural network with upsampling and ReLU
        """
        # Feed conditioning input into a Dense unit
        x = Concatenate()([input_G, conditioning_label])
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(128*self.filters*7*7)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape((7, 7, 128*self.filters), input_shape=(128*self.filters*7*7,))(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64*self.filters, (5, 5), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(1, (5, 5), padding='same')(x)
        self.output_G = Activation('tanh')(x)

        # Assemble the model
        model = Model([input_G, conditioning_label], self.output_G)
        model.summary()
        return model

    def discriminator(self, input_D, conditioning_label):
        """ CGAN Discriminator, small neural network with upsampling
        """
        # Concatenate the units and feed to the shared branch
        y = Dense(28 * 28)(conditioning_label)
        y = Reshape((28, 28, 1))(y)
        x = concatenate([input_D, y])
        x = Conv2D(64*self.filters, (5, 5), padding='same')(x)
        for i in range(self.depth):
            x = BatchNormalization()(x)
            x = ELU()(x)
            x = Conv2D(128*self.filters*(i+1), (5, 5), padding='same')(x)
        x = BatchNormalization()(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(4*self.filters)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = ELU()(x)
        output_D = Dense(1, activation ='sigmoid')(x)

        # Assemble the model
        model = Model([input_D, conditioning_label], output_D)
        model.summary()
        return model

    # Labels have to be in one hot encoding!
    def generate(self, labels):
        if self.G is None:
            raise ValueError('Training not yet implemented')
        X_g = np.random.uniform(-1, 1, (len(labels), self.noise_dim))
        return self.G.predict([X_g,labels], verbose=0)
