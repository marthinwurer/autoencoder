"""
This file loads a directory of images into a keras image dataset

based off of https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
https://stackoverflow.com/questions/43239897/create-readable-image-dataset-for-training-in-keras
"""
from os import listdir
from os.path import isfile, join
from sys import argv

# figure out the path stuff before we actually start loading things so we don't have to
# wait for tensorflow to boot

base_path = argv[1] + '/'
print(base_path)

train_data_dir = base_path + '/train'
validation_data_dir = base_path + '/validation'

train_files = [file for file in listdir(train_data_dir) if isfile(join(train_data_dir, file))]

print(train_files)



from keras import backend as K
from scipy import misc

# dimensions of our images.
img_width, img_height = 128, 150

base_path = argv[1]
print(base_path)

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

print(K.image_data_format())

# load all the files

train_files = [file for file in listdir(train_data_dir) if isfile(join(train_data_dir, file))]

print(train_files)

