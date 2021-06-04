import numpy as np
import tensorflow as tf

# Tensors are the basic data structure in TensorFlow which store data in any number of dimensions, similar to multi dimensional arrays in NumPy.
#   Constants - nodes without input (immutable),
#   Variables - mutable nodes,
#   or Placeholders - "promise" that value will be assigned (used for inputs)


# The most basic code possible:
## --
## Define a variable
#a = tf.Variable(2.0, name='a')
## Define a constant
#b = tf.constant([1., 2., 3., 4.], name='b')
## Combine tensors to create a new tensor
#c = (a + 1)*b
## --

# Working with Arrays:
## --
## Let's do this exercise again with more array operations:
#b = tf.Variable(np.arange(0, 10), name='b')
#c = tf.Variable(1.0, name='c')
## To add, we need to convert b to float (since the original values cast to int32)
#d = tf.cast(b, tf.float32) + c
## --

# Lets get into some neural networks
## --
## Nab a dataset
## Note on the data dimensions:
##  x_train: (60,000 x 28 x 28)
##  y_train: (60,000)
##  x_test: (10,000 x 28 x 28)
##  y_test: (10,000)
## Note: x_data is scaled to be [0, 1]
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = tf.Variable(x_train)
x_test = tf.Variable(x_test)
## Create a function for batch extraction of training data:
def get_batch(x_data, y_data, batch_size):
    idxs = np.random.randint(0, len(y_data), batch_size)
    print(x_data[idxs.to_list(),:,:])
    print(y_data[idxs])
    return x_data[idxs,:,:], y_data[idxs]
## Setup simulation parameters:
epochs = 10
batch_size = 100
## Create weights and biases based on a static, one-hidden-layer structure
## Note that the input is 28 x 28 = 784
W1 = tf.Variable(tf.random.normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random.normal([300]), name='b1')
W2 = tf.Variable(tf.random.normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random.normal([10]), name='b2')
## Having initialized our arrays, create a neural network function with tf operations as follows:
def nn_model(x_input, W1, b1, W2, b2):
    x_input = tf.reshape(x_input, (x_input.shape[0], -1)) # Flatten
    x = tf.add(tf.matmul(tf.cast(x_input, tf.float32), W1), b1) #Layer 1
    x = tf.nn.relu(x) # Activate Layer 1
    logits = tf.add(tf.matmul(x, W2), b2) # Layer 2
    return logits # Return unactivated output because the logit is needed for the loss function
## Having created a model, setup a loss function as follows:
def loss_fn(logits, labels):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,                                                          logits=logits))
    return cross_entropy
## Finally Define the Optimizer:
optimizer = tf.keras.optimizers.Adam()
## Train the network (copy and paste code):
total_batch = int(len(y_train) / batch_size)
for epoch in range(epochs):
    avg_loss = 0
    for i in range(total_batch):
        batch_x, batch_y = get_batch(x_train, y_train, batch_size=batch_size)
        batch_x = tf.Variable(batch_x)
        batch_y = tf.Variable(batch_y)
        batch_y = tf.one_hot(batch_y, 10)
        with tf.GradientTape() as tape:
            logits = nn_model(batch_x, W1, b1, W2, b2)
            loss = loss_fn(logits, batch_y)
        gradients = tape.gradient(loss, [W1, b1, W2, b2])
        optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))
        avg_loss += loss / total_batch
    test_logits = nn_model(x_test, W1, b1, W2, b2)
    max_idxs = tf.argmax(test_logits, axis=1)
    test_acc = np.sum(max_idxs.numpy() == y_test) / len(y_test)
    print(f"Epoch: {epoch + 1}, loss={avg_loss:.3f}, test set      accuracy={test_acc*100:.3f}%")
print("\nTraining complete!")
