from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.optimizers import Adam
from gan import GAN
from tqdm import tqdm
from keras.utils import plot_model
import json
import numpy as np

class DCGAN(GAN):
    """ Deep Convolutional GAN, as per https://arxiv.org/abs/1511.06434
    """

    def __init__(self, name, depth, filters):
        super(DCGAN, self).__init__(name)
        self.depth = depth
        self.filters = filters
        self.build_model()

    def build_model(self):
        self.G = self.generator()
        self.D = self.discriminator()
        self.D.trainable = True
        self.D.compile(Adam(self.lr_d, beta_1 = 0.5), "binary_crossentropy", metrics = ["accuracy"])
        self.D.trainable = False
        self.m = Sequential([self.G, self.D])
        self.m.compile(Adam(self.lr_g, beta_1 = 0.5), "binary_crossentropy", metrics = ["accuracy"])
        plot_model(self.G, to_file="../../Figures/CW2/{}/G_{}.png".format(self.name,self.name))
        plot_model(self.D, to_file="../../Figures/CW2/{}/D_{}.png".format(self.name,self.name))

    def train(self, X_train, y_train=None, epochs=10, sample_interval=1, batch_size=128, save=True):
        """ Train DCGAN:
            - Train D to discriminate G results
            - Train G to fool D (D is frozen)
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
                reference_imgs = X_train[reference_idx]

            for i in tqdm(range(X_train.shape[0] // self.batch_size)):
                # Select a random half batch of images
                idx = None
                X_d_true = None


                if self.virtual_batch_norm:
                    idx = np.random.randint(0, X_train.shape[0], self.batch_size//2)
                    imgs_random = X_train[idx]

                    # Add them together
                    imgs = np.concatenate((imgs_random, reference_imgs), axis=0)


                    mean = np.mean(imgs, axis=0)
                    std = np.std(imgs, axis = 0)
                    std[std==0] = 1
                    imgs = (imgs - mean)/std

                    # Reshuffle the images
                    p = np.random.permutation(len(imgs))
                    X_d_true = imgs[p]

                else:
                    idx = np.random.randint(0, X_train.shape[0], self.batch_size)
                    X_d_true = X_train[idx]


                """ Generate fake and real data to train D
                """

                X_g = np.array([np.random.normal(0,0.5,100) for _ in range(self.batch_size)])
                X_d_gen = self.G.predict(X_g, verbose=0)

                # Train D
                d_loss=self.D.train_on_batch(X_d_true, y_d_true)
                d_loss+= self.D.train_on_batch(X_d_gen, y_d_gen)

                prediction_true = self.D.predict_on_batch(X_d_true)
                prediction_gen = self.D.predict_on_batch(X_d_gen)

                d_accuracy = 0
                for i in range(len(prediction_true)):
                    if int(prediction_true[i]) == y_d_true[i]:
                        d_accuracy+=1

                for i in range(len(prediction_gen)):
                    if int(prediction_gen[i]) == y_d_gen[i]:
                        d_accuracy+=1
                d_accuracy/=(len(prediction_true)+ len(prediction_gen))

                g_loss = self.m.train_on_batch(X_g, y_g)

                discriminator_loss_epoch.append(d_loss)
                discriminator_accuracy_epoch.append(d_accuracy)
                generator_loss_epoch.append(g_loss)


            self.discriminator_loss.append(np.asscalar(np.array(discriminator_loss_epoch).mean()))
            self.discriminator_accuracy.append(np.asscalar(np.array(discriminator_accuracy_epoch).mean()))
            self.generator_loss.append(np.asscalar(np.array(generator_loss_epoch).mean()))

            print ("[D loss: %f, acc.: %.2f%%] [G loss: %f]" % (self.discriminator_loss[epoch],
                                                            self.discriminator_accuracy[epoch],
                                                            self.generator_loss[epoch]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_imgs(epoch)

        if save:
            self.G.save("models/"+self.name+"_generator.h5")
            self.D.save("models/"+self.name+"_discriminator.h5")
            self.m.save("models/"+self.name+"_complete.h5")
            history = {"discriminator_loss": self.discriminator_loss,
                       "discriminator_accuracy": self.discriminator_accuracy,
                       "generator_loss": self.generator_loss}
            with open("models/"+self.name+"_history.json", 'w') as outfile:
                json.dump(history, outfile)
        return history


    def generator(self):
        """ DCGAN Generator, small neural network with Upsampling and ReLU
        """
        model = Sequential()
        model.add(Dense(input_dim=100, units=1024))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Dense(128*self.filters*7*7))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(Reshape((7, 7, 128*self.filters), input_shape=(128*self.filters*7*7,)))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(64*self.filters, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(1, (5, 5), padding='same'))
        model.add(Activation('tanh'))
        print(model.summary())
        return model

    def discriminator(self):
        """ DCGAN Discriminator, small neural network with upsampling
        """
        model = Sequential()
        model.add(Conv2D(64*self.filters, (5, 5), padding='same', input_shape=self.img_shape))
        for i in range(self.depth):
            model.add(BatchNormalization())
            model.add(ELU())
            model.add(Conv2D(128*self.filters*(i+1), (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(ELU())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(4*self.filters))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(ELU())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        print(model.summary())
        return model
