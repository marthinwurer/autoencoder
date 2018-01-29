from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.datasets import cifar10
import make_dataset

#                      height, width, channels
input_img = Input(shape=(128, 256, 3))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (16, 8, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.summary()

# opt = Adam()
opt = SGD(lr=0.01, clipvalue=0.5)



autoencoder.compile(optimizer=opt, loss='binary_crossentropy')


# set up the data
(x_train, _), (x_test, _) = make_dataset.load_data(".")
x_train = x_train.astype('float32')/255. # Make sure that the data is between 0 and 1
x_test = x_test.astype('float32')/255.

print(x_train.shape)


autoencoder.fit(x_train, x_train,
                epochs=500,
                batch_size=4,
                shuffle=True,
                validation_data=(x_test, x_test))



import matplotlib.pyplot as plt
plt.imshow(autoencoder.predict(x_test)[0])
plt.show()

