import argparse
import make_dataset

parser = argparse.ArgumentParser()
parser.add_argument("base", help="base directory where dataset data is found")
args = parser.parse_args()
base = args.base + "/"

# set up the data
(x_train, y_train), (x_test, y_test) = make_dataset.load_data(args.base)
x_train = x_train.astype('float32')/255. # Make sure that the data is between 0 and 1
x_test = x_test.astype('float32')/255.

print("x_train", x_train.shape)
print("y_train", y_train.shape)

# keras stuff
import keras
from keras.layers import Input, Dense, Conv2D, Dropout, BatchNormalization, Flatten
from keras.models import Model
from keras.optimizers import Adam, SGD

input_img = Input(shape=x_test.shape[1:])

dropout_value = 0.5

# initial layer
x = Conv2D(16, 3, activation='relu', padding='same')(input_img)
x = BatchNormalization()(x)
x = Dropout(dropout_value)(x)

# shrink it down
x = Conv2D(32, 3, strides=2, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Dropout(dropout_value)(x)

x = Conv2D(32, 3, strides=2, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Dropout(dropout_value)(x)

x = Conv2D(32, 3, strides=2, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Dropout(dropout_value)(x)

x = Conv2D(32, 3, strides=2, activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Dropout(dropout_value)(x)

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(dropout_value)(x)
classes = Dense(y_test.shape[1], activation='softmax', name='classes')(x)


classifier = Model(input_img, classes)
classifier.summary()

opt = Adam()
# opt = SGD(lr=0.01, momentum=.9, clipvalue=0.5)

classifier.compile(optimizer=opt,
                   metrics=['mse', 'categorical_accuracy'],
                   loss='categorical_crossentropy')

classifier.fit(x_train, y_train,
                epochs=100,
                batch_size=32,
                shuffle=True,
                validation_data=(x_test, y_test))

classifier.save(base + "classifier.h5")





