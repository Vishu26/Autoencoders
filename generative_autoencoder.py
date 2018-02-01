from keras.layers import Conv2D, Flatten, MaxPool2D, Dense
from keras.layers import UpSampling2D, Conv2DTranspose, Reshape
from keras.models import Input, Model, Sequential, load_model
from keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
import numpy as np
import h5py

X, y = load_data()
X = X[0]
z = X


X = X[:, np.newaxis, :, :]

x_train = Input(shape=(1, 28, 28))


encoder = Sequential()
encoder.add(Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu', input_shape=(1, 28, 28)))
encoder.add(MaxPool2D((2, 2), 2, dim_ordering='th'))
encoder.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'))
encoder.add(MaxPool2D((2, 2), 2, dim_ordering='th'))
encoder.add(Flatten())
encoder.add(Dense(64, activation='relu'))

encoded = encoder(x_train)

decoder = Sequential()
decoder.add(Reshape(target_shape=(-1, 8, 8), input_shape=(64,)))
decoder.add(Conv2DTranspose(32, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2DTranspose(16, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Flatten())
decoder.add(Dense(64, activation='relu'))
decoder.add(Dense(784, activation='sigmoid'))
decoder.add(Reshape(target_shape=(28, 28)))
encoder.summary()
decoder.summary()
decoded = decoder(encoded)

autoencoder = Model(x_train, decoded)

autoencoder.summary()

encoded_input = Input(shape=(64, ))

#decoder_layer = autoencoder.layers[-1]

#dec = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(X, z, epochs=10, validation_split=0.2, batch_size=1024)



