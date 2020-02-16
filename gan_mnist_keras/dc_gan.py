# https://datascienceschool.net/view-notebook/7788014b90364dd5ba9dc76f35b4cd7d/

import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
from keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                          Flatten, Input, Reshape, ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam

generator_ = Sequential()
generator_.add(Dense(128 * 7 * 7, activation="relu", input_shape=(100,)))
generator_.add(Reshape((7, 7, 128)))
generator_.add(BatchNormalization(momentum=0.8))
generator_.add(UpSampling2D())
generator_.add(Conv2D(128, kernel_size=3, padding="same"))
generator_.add(Activation("relu"))
generator_.add(BatchNormalization(momentum=0.8))
generator_.add(UpSampling2D())
generator_.add(Conv2D(64, kernel_size=3, padding="same"))
generator_.add(Activation("relu"))
generator_.add(BatchNormalization(momentum=0.8))
generator_.add(Conv2D(1, kernel_size=3, padding="same"))
generator_.add(Activation("tanh"))

noise_input = Input(shape=(100,), name="noise_input")
generator = Model(noise_input, generator_(noise_input), name="generator")

generator_.summary()
generator.summary()

generator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))


noise_data = np.random.normal(0, 1, (32, 100))
generated_images = 0.5 * generator.predict(noise_data) + 0.5
generated_images.shape


def show_images(generated_images, n=4, m=8, figsize=(9, 5)):
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=1, bottom=0, hspace=0, wspace=0.05)
    for i in range(n):
        for j in range(m):
            k = i * m + j
            ax = fig.add_subplot(n, m, i * m + j + 1)
            ax.imshow(generated_images[k][:, :, 0], cmap=plt.cm.bone)
            ax.grid(False)
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
    plt.tight_layout()
    plt.show()


show_images(generated_images)


discriminator_ = Sequential()
discriminator_.add(Conv2D(32, kernel_size=3, strides=2,
                          input_shape=(28, 28, 1), padding="same"))
discriminator_.add(LeakyReLU(alpha=0.2))
discriminator_.add(Dropout(0.25))
discriminator_.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
discriminator_.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
discriminator_.add(LeakyReLU(alpha=0.2))
discriminator_.add(Dropout(0.25))
discriminator_.add(BatchNormalization(momentum=0.8))
discriminator_.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
discriminator_.add(LeakyReLU(alpha=0.2))
discriminator_.add(Dropout(0.25))
discriminator_.add(BatchNormalization(momentum=0.8))
discriminator_.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
discriminator_.add(LeakyReLU(alpha=0.2))
discriminator_.add(Dropout(0.25))
discriminator_.add(Flatten())
discriminator_.add(Dense(1, activation='sigmoid'))

image_input = Input(shape=(28, 28, 1), name="image_input")
discriminator = Model(image_input, discriminator_(
    image_input), name="discriminator")

discriminator_.summary()
discriminator.summary()

discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])


noise_input2 = Input(shape=(100,), name="noise_input2")
combined = Model(noise_input2, discriminator(generator(noise_input2)))
combined.summary()

combined.compile(loss='binary_crossentropy',
                 optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])


(X_train, _), (_, _) = mnist.load_data()

# Rescale -1 to 1
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = np.expand_dims(X_train, axis=3)

batch_size = 128
half_batch = int(batch_size / 2)


def train(epochs, print_step=10):
    history = []
    for epoch in range(epochs):

        # discriminator 트레이닝 단계
        #######################################################################3

        # 데이터 절반은 실제 이미지, 절반은 generator가 생성한 가짜 이미지
        real_images = X_train[np.random.randint(0, X_train.shape[0], half_batch)]
        y_real = np.ones((half_batch, 1))
        generated_images = generator.predict(np.random.normal(0, 1, (half_batch, 100)))
        y_generated = np.zeros((half_batch, 1))

        # discriminator가 실제 이미지와 가짜 이미지를 구별하도록 discriminator를 트레이닝
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_images, y_real)
        d_loss_fake = discriminator.train_on_batch(generated_images, y_generated)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # generator 트레이닝 단계
        #######################################################################3

        # 전부 generator가 생성한 가짜 이미지를 사용.
        # discriminator가 구별하지 못하도록 generator를 트레이닝
        noise = np.random.normal(0, 1, (batch_size, 100))
        discriminator.trainable = False
        g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

        # 기록
        record = (epoch, d_loss[0], 100 * d_loss[1], g_loss[0], 100 * g_loss[1])
        history.append(record)
        if epoch % print_step == 0:
            print("%5d [D loss: %.3f, acc.: %.2f%%] [G loss: %.3f, acc.: %.2f%%]" % record)


history100 = train(100)
show_images(0.5 * generator.predict(noise_data) + 0.5)


def save_models(epoch):
    generator.save("dcgan_generator_{}.hdf5".format(epoch))
    discriminator.save("dcgan_discriminator_{}.hdf5".format(epoch))
    combined.save("dcgan_combined_{}.hdf5".format(epoch))


history1000 = train(1000, 100)
show_images(0.5 * generator.predict(noise_data) + 0.5)
save_models(1000)

history2000 = train(1000, 100)
show_images(0.5 * generator.predict(noise_data) + 0.5)
save_models(2000)

history3000 = train(1000, 100)
show_images(0.5 * generator.predict(noise_data) + 0.5)
save_models(3000)

history4000 = train(1000, 100)
show_images(0.5 * generator.predict(noise_data) + 0.5)
save_models(4000)

history5000 = train(1000, 100)
show_images(0.5 * generator.predict(noise_data) + 0.5)
save_models(5000)

history10000 = train(5000, 100)
show_images(0.5 * generator.predict(noise_data) + 0.5)
save_models(10000)
