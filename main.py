import argparse

import make_dataset

parser = argparse.ArgumentParser()
parser.add_argument("base", help="base directory where dataset data is found")
# parser.add_argument("--height", type=int)
# parser.add_argument("--width", type=int)
args = parser.parse_args()
base = args.base + "/"

# set up the data
(x_train, _), (x_test, _) = make_dataset.load_data(args.base)
x_train = x_train.astype('float32')/255. # Make sure that the data is between 0 and 1
x_test = x_test.astype('float32')/255.

print(x_train.shape)

# set up the keras stuff
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, AveragePooling2D, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.datasets import cifar10
#                      height, width, channels
input_img = Input(shape=x_test.shape[1:])  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = AveragePooling2D((2, 2), padding='same')(x)
# x = BatchNormalization()(x)
x = Dropout(0.1)(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = AveragePooling2D((2, 2), padding='same')(x)
# x = BatchNormalization()(x)
x = Dropout(0.1)(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (16, 8, 8) i.e. 128-dimensional

# x = BatchNormalization()(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = Dropout(0.1)(x)
# x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Dropout(0.1)(x)
# x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Dropout(0.1)(x)
# x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()

opt = Adam()
# opt = SGD(lr=0.01, momentum=.9, clipvalue=0.5)



autoencoder.compile(optimizer=opt, loss='binary_crossentropy')




autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=32,
                shuffle=True,
                validation_data=(x_test, x_test))


autoencoder.save(base + "autoencoder.h5")


import matplotlib.pyplot as plt
plt.imshow(autoencoder.predict(x_test)[0])
plt.show()

