# Change system directory (neural network code in parent directory)
import sys
sys.path.append("..")

# Imports, including custom neural network
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from custom_neural_network import neural_network

# Create Gaussian Data Clusters for Training
n = 1000 # number of samples per cluster
X1 = np.random.multivariate_normal([-10,0], [[1,0],[0,1]], n)             # Cluster 1
X2 = np.random.multivariate_normal([0,10], [[1,0.25],[0.25,1]], n)       # Cluster 2
X3 = np.random.multivariate_normal([5,2.5], [[1,0.5],[0.5,1]], n)       # Cluster 3
XT = np.concatenate((X1, X2, X3), 0)

YT = np.zeros([XT.shape[0], 3])
YT[0:n, 0] = 1
YT[n+1:2*n, 1] = 1
YT[2*n+1:3*n, 2] = 1

X_train, X_test, Y_train, Y_test = train_test_split(XT, YT, test_size=0.2)

X = np.array(X_train).T
Y = np.array(Y_train).T

# Train the Neural Network
myNet = neural_network([2, 256, 128, 3])
myNet.training_batch_size = 1
myNet.learning_rates = 0.0005 * np.ones(6)
myNet.train_network(X, Y, 30000)

# Predict Until Program Exit
X = np.array(X_test).T
Y = np.array(Y_test).T
while True:
    i = np.random.randint(0, X.shape[1], 1)
    Y_hat = myNet.classify_data(X[:,i])
    print(Y_hat)
    print(Y[:,i])
    input()

'''
# Copy / Paste Above to Show Data
plt.plot(X1[:, 0], X1[:, 1], 'b.')
plt.plot(X2[:, 0], X2[:, 1], 'r.')
plt.plot(X3[:, 0], X3[:, 1], 'g.')
plt.show()
'''