# Change system directory (neural network code in parent directory)
import sys
sys.path.append("..")

# Imports, including custom neural network
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from custom_neural_network import neural_network

# Import Iris Data for Training
XT, YT = load_iris(return_X_y = True)
newYT = np.zeros([len(YT), 3])
for i, y in enumerate(YT):
    newYT[i][y] = 1

X_train, X_test, Y_train, Y_test = train_test_split(XT, newYT, test_size=0.2)

X = np.array(X_train).T
Y = np.array(Y_train).T
print(X.shape)
print(Y.shape)
input()

# Train the Neural Network
myNet = neural_network([4, 128, 64, 3])
myNet.training_batch_size = 1
myNet.learning_rates = 0.001 * np.ones(4)
myNet.train_network(X, Y, 100000)

# Predict Until Program Exit
X = np.array(X_test).T
Y = np.array(Y_test).T
while True:
    i = np.random.randint(0, X.shape[1], 1)
    Y_hat = myNet.classify_data(X[:,i])
    print(Y_hat)
    print(Y[:,i])
    input()
