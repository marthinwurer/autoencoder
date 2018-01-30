
# before loading keras and tensorflow, check the inputs
import sys
image_file = sys.argv[1]


from keras.models import load_model
from scipy import misc
from skimage.transform import resize
import make_dataset
import numpy as np
import matplotlib.pyplot as plt

# load the model
model = load_model("./autoencoder.h5")

print(model.input_shape)

image = make_dataset.load_image(image_file, (model.input_shape[1], model.input_shape[2]))

print(image.shape)
image = image.astype('float32')/255. # Make sure that the data is between 0 and 1

out = model.predict(np.asarray([image]))
print(out.shape)

out = out[0]


# make a new plot, and get the figure from it.
fig = plt.figure()
axe = fig.add_subplot(1, 2, 2) # vertical, horizontal, index
axe.imshow(image)
axe = fig.add_subplot(1, 2, 1)
axe.imshow(out)

# show the plot
plt.show()

