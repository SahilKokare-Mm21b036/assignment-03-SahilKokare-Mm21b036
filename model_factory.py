import numpy as np
import struct
import pickle
from utils import *
from dense_neural_class import *


# Reads the MNIST image file and returns a NumPy array with the images.
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
    return images

# Reads the MNIST label file and returns a NumPy array with the labels.
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


# Load train set
train_images = load_mnist_images('./mnist/train-images.idx3-ubyte')
train_labels = load_mnist_labels('./mnist/train-labels.idx1-ubyte')
# Load test set
test_images = load_mnist_images('./mnist/t10k-images.idx3-ubyte')
test_labels = load_mnist_labels('./mnist/t10k-labels.idx1-ubyte')

# Check shapes
print(f"Training images: {train_images.shape}")  # Ex.: (60000, 28, 28)
print(f"Training labels: {train_labels.shape}")
print(f"Testing images: {test_images.shape}")# Ex.: (60000,)
print(f"Testing labels: {test_labels.shape}")


# Putting the data in a best-known format.
# Train set
X = train_images
X = X.reshape(-1,28*28)
Y = train_labels
Y = Y.reshape(-1,1)

# Test set
X_test = test_images
X_test = X_test.reshape(-1,28*28)
Y_test = test_labels
Y_test = Y_test.reshape(-1,1)


# Instantiation of the neural network as model2.
model2 = Dense_Neural_Diy(input_size=784, hidden_layer1_size=50, hidden_layer2_size=20 , output_size=10)

model2.fit(X,Y,learning_rate=0.005, epochs=11, batch_size=60000 )
model2.improve_train(X,Y, learning_rate=0.005, epochs=61, batch_size=40)


save_model('final_model', model2)
