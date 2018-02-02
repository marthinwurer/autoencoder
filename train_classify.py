import argparse
import make_dataset

parser = argparse.ArgumentParser()
parser.add_argument("base", help="base directory where dataset data is found")
# parser.add_argument("--height", type=int)
# parser.add_argument("--width", type=int)
args = parser.parse_args()
base = args.base + "/"

# set up the data
(x_train, y_train), (x_test, y_test) = make_dataset.load_data(args.base)
x_train = x_train.astype('float32')/255. # Make sure that the data is between 0 and 1
x_test = x_test.astype('float32')/255.






