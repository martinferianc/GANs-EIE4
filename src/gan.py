import numpy as np
import matplotlib.pyplot as plt
import os
import json
from keras.utils import to_categorical
from keras.models import load_model



class GAN(object):
    """ Generic GAN Class
    """

    def __init__(self, name):
        self.img_shape = (28, 28, 1)
        self.noise_dim = 100
        self.class_dim = 10
        self.lr = 2e-4
        self.lr_d =  self.lr
        self.lr_g = self.lr*10
        self.filters = 1
        self.depth = 1
        self.name = name
        self.virtual_batch_norm = False
        self.label_smoothing = False
        self.discriminator_loss = []
        self.discriminator_accuracy = []
        self.generator_loss = []
        self.inception_score = []
        self.noise = np.random.uniform(-1, 1, (2 * 5, self.noise_dim))
        self.sampled_labels = np.arange(0, 10).reshape(-1, 1)
        self.sampled_labels = to_categorical(self.sampled_labels, self.class_dim)

        if not os.path.isdir("../../Figures/CW2/{}".format(self.name)):
            os.makedirs("../../Figures/CW2/{}".format(self.name))


    def train(self, X_train, y_train, epochs, batch_size=128, sample_interval=1, save = False, x_test=None, y_test=None):
        raise ValueError('Training not yet implemented')


    def sample_images(self, epoch):
        r, c = 2, 5

        gen_imgs = self.G.predict([self.noise,self.sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        fig.tight_layout()
        fig.suptitle('Class samples for network: {}'.format(self.name))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title("Digit: %d" % np.argmax(self.sampled_labels[cnt]))
                axs[i,j].axis('off')
                cnt += 1

        fig.savefig("../../Figures/CW2/{}/sample_{}_{}.png".format(self.name,self.name, epoch))
        plt.close()

    def save_imgs(self, epoch):
        r, c = 2, 5
        gen_imgs = self.G.predict(self.noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        fig.tight_layout()
        fig.suptitle('Samples for network: {}'.format(self.name))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                axs[i,j].set_title("Sample: {}".format(cnt+1))
                cnt += 1

        fig.savefig("../../Figures/CW2/{}/sample_labeled_{}_{}.png".format(self.name,self.name, epoch))
        plt.close()


    def load_model(self, model_path, only_history=True):
        if only_history is False:
            self.m = load_model(model_path+"_complete.h5")
            self.G = load_model(model_path+"_generator.h5")
            self.D= load_model(model_path+"_discriminator.h5")
        with open(model_path+"_history.json") as f:
            history = json.load(f)
            self.discriminator_loss = history["discriminator_loss"]
            self.discriminator_accuracy = history["discriminator_accuracy"]
            self.generator_loss = history["generator_loss"]
            if "DC" not in self.name:
                self.inception_score = history["inception_score"]
            return history
