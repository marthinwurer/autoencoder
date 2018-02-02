"""

"""
from scipy import ndimage
from scipy import misc
import numpy

import matplotlib.pyplot as plt
import sys
import os
import argparse

from skimage.transform import resize

# from keras.datasets import *
#
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(type(x_train), x_train.shape, x_train.dtype)
# print(type(y_train), y_train.shape, y_train.dtype)

TRAIN_DATA_FILE = "train_data.npy"
TRAIN_LABEL_FILE = "train_labels.npy"
TEST_DATA_FILE = "test_data.npy"
TEST_LABEL_FILE = "test_labels.npy"
NAMES_FILE = "names.csv"

def load_data(path):
    # load the arrays
    train_data = numpy.load(path+"/"+TRAIN_DATA_FILE)
    train_labels = numpy.load(path+"/"+TRAIN_LABEL_FILE)
    test_data = numpy.load(path+"/"+TEST_DATA_FILE)
    test_labels = numpy.load(path+"/"+TEST_LABEL_FILE)

    # load the names file
    names = {}
    with open(path+"/"+NAMES_FILE) as f:
        for line in f:
            line = line.split(",")
            names[line[0].strip()] = int(line[1].strip())

    print(train_data.shape)
    print(test_data.shape)
    print(names)

    # expand the labels into one-hot vectors
    test_label_vectors = []
    train_label_vectors = []
    width = len(names)
    for label in test_labels:
        vec = numpy.zeros(width)
        vec[label] = 1
        test_label_vectors.append(vec)
    for label in train_labels:
        vec = numpy.zeros(width)
        vec[label] = 1
        train_label_vectors.append(vec)
    test_labels = numpy.asarray(test_label_vectors)
    train_labels = numpy.asarray(train_label_vectors)


    return (train_data, train_labels), (test_data, test_labels)

def load_image(path:str, size: tuple=None):
    image = misc.imread(path)

    if len(image.shape) == 2:
        # only one channel, reshape to have a single dimensional channel
        # image = numpy.reshape(image, image.shape + (1,))
        # stack it thrice
        image = numpy.stack((image,)*3, axis=-1)
        print(image.shape)
    # if there are more than 3 channels, remove the alpha channel
    elif image.shape[2] > 3:
        # remove the alpha channel
        image = numpy.delete(image, [3], axis=2)
    # resize
    if size is not None:
        image = resize(image, size, preserve_range=True)#, anti_aliasing=True)
    return image


def load_images(name: str, directories: set, names: dict=None, size: tuple=None):
    """Load all the png images, remove the alpha channel, and join them all into numpy arrays

    Args:
        name:
        directories:
        names:

    Returns:

    """
    images = []
    labels = []
    if names is None:
        names = {} # name -> activation vector index

    for index, directory in enumerate(directories):
        image_files = os.listdir(name+"/"+directory)
        if not directory in names:
            names[directory] = index
        for file in image_files:
            # print(file)
            image = load_image(name+"/"+directory+"/"+file, size=size)

            # store in array
            images.append(image)
            # add the label index
            labels.append(index)


        # show last image
        print(type(image), image.shape, image.dtype)
        # plt.imshow(image)
        # plt.show()
        # print(image)

    images = numpy.asarray(images)
    labels = numpy.asarray(labels, numpy.uint8)


    print(names)
    return images, labels, names




def main( argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("base", help="base directory where dataset data is found")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)

    args = parser.parse_args()

    # get the base directory from the first arg
    base = args.base
    test_dir = base+"/test/"
    train_dir = base+"/train/"

    files = os.listdir(base)

    print(files)

    # make sure that test and train are in base directory
    if not( "test" in files and "train" in files):
        raise UserWarning("No 'test' or 'train' directories")
    # make sure that both of their subdirs match
    train_dirs = set(os.listdir(train_dir))
    print(train_dirs)

    # make sure that the test dirs match
    test_dirs = set(os.listdir(test_dir))
    if len(test_dirs) != len(train_dirs):
        raise UserWarning("Train and Test directories have different numbers of directories")

    for directory in test_dirs:
        if directory not in train_dirs:
            raise UserWarning(directory + " not a training directory")

    print("Directories match, generating train files")
    images, labels, names = load_images(train_dir, train_dirs)#, size=(64, 128))

    print(type(images), images.shape, images.dtype)

    # save images
    numpy.save("train_data.npy", images)
    numpy.save("train_labels.npy", labels)

    # then do the test files
    images, labels, names = load_images(test_dir, train_dirs)
    print(type(images), images.shape, images.dtype)

    numpy.save("test_data.npy", images)
    numpy.save("test_labels.npy", labels)

    # dump the names file
    with open(NAMES_FILE, "w") as f:
        for k, v in names.items():
            f.write(k + ", " + str(v) + "\n")

    plt.imshow(images[0])
    plt.show()
    print(labels[0])
    print(labels)


    (train_images, train_labels), (test_images, test_labels) = load_data(base)
    print(test_labels[0])

    plt.imshow(test_images[0])
    plt.show()
































if __name__=="__main__":
    main(sys.argv)

