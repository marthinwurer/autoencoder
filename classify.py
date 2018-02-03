import argparse

from keras.models import load_model
import make_dataset
import numpy as np
import matplotlib.pyplot as plt

# load the model

import tkinter as tk
from tkinter import filedialog

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base", help="base directory where dataset data is found")
    args = parser.parse_args()
    base = args.base + "/"

    # set up the data
    (x_train, y_train), (x_test, y_test) = make_dataset.load_data(args.base)
    x_train = x_train.astype('float32')/255. # Make sure that the data is between 0 and 1
    x_test = x_test.astype('float32')/255.
    classes = y_test.shape[1]


    model = load_model(base+"classifier.h5")

    print(model.input_shape)


    test_out = model.predict(x_test)
    train_out = model.predict(x_train)
    print(test_out.shape)

    # build a confusion matrix
    confusion_test = np.zeros((classes,classes))
    missed = []
    for index, item in enumerate(test_out):
        current = np.argmax(y_test[index])
        value = np.argmax(item)
        confusion_test[current][value] += 1
        if current != value:
            missed.append(index)

    # train matrix
    confusion_train = np.zeros((classes,classes))
    for index, item in enumerate(train_out):
        current = np.argmax(y_train[index])
        value = np.argmax(item)
        confusion_train[current][value] += 1

    print("test error:",len(missed), len(test_out), "=", len(missed)/ len(test_out))

    # make a new plot, and get the figure from it.
    fig = plt.figure()
    axe = fig.add_subplot(1, 2, 1) # vertical, horizontal, index
    axe.imshow(confusion_test)
    axe = fig.add_subplot(1, 2, 2)
    axe.imshow(confusion_train)

    # show the plot
    plt.show()

if __name__=="__main__":
    main()
