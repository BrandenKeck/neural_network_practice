# Change system directory
import sys
sys.path.append("..")

# Imports
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pytorch_example_network import neural_network

# Establish Data
XT, YT = load_iris(return_X_y = True)
newYT = np.zeros([len(YT), 3])
for i, y in enumerate(YT):
    newYT[i][y] = 1
X_train, X_test, Y_train, Y_test = train_test_split(XT, newYT, test_size=0.2)

# Train Network
X = np.array(X_train)
Y = np.array(Y_train)
myNet = neural_network()
myNet.train_network(X, Y)

# Test Predictions
X = np.array(X_test)
Y = np.array(Y_test)
while True:
    i = np.random.randint(0, X.shape[0], 1)
    Y_hat = myNet.predict_network(X[i, :])
    print(Y_hat)
    print(Y[i, :])
    input()

