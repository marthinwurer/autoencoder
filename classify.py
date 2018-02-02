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


    out = model.predict(x_test)
    print(out.shape)

    # build a confusion matrix
    confusion = np.zeros((classes,classes))
    missed = []
    for index, out in enumerate(out):
        current = np.argmax(y_test[index])
        value = np.argmax(out)
        confusion[current][value] += 1
        if current != value:
            missed.append(index)


    print("error",len(missed), len(out))


    # make a new plot, and get the figure from it.
    fig = plt.figure()
    axe = fig.add_subplot(1, 2, 1) # vertical, horizontal, index
    axe.imshow(confusion)
    axe = fig.add_subplot(1, 2, 2)
    axe.imshow(confusion)

    # show the plot
    plt.show()

if __name__=="__main__":
    main()
