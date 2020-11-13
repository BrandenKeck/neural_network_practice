# Standard Imports
import numpy as np

# Custom Imports
from diagnostics import network_diagnostics

# Neural Network Class for Feature Approximation in Reinforcement Learning
class neural_network():

    def __init__(self, layersizes):

        # Track layersizes as a numpy array
        self.layersizes = np.array([int(i) for i in layersizes])

        # Initialize Node Arrays
        self.a = []
        self.z = []
        self.y_hat = None

        # Initialize Weight Arrays
        self.w = []
        self.b = []
        for i in np.arange(1, len(self.layersizes)):
            self.w.append(0.01 * np.random.randn(self.layersizes[i], self.layersizes[i-1]))
            self.b.append(np.zeros(self.layersizes[i]))

        # Misc. Network Settings
        self.training_batch_size = 1
        self.learning_rates = 0.01 * np.ones(len(self.w))
        self.leaky_relu_rates = 0.01 * np.ones(len(self.w))
        self.huber_cost_delta = 5

        # Initialize Cost Function Settings
        # DEFAULT: Huber Cost Function
        self.use_huber_cost = False
        self.use_hellinger_cost = False
        self.use_quadratic_cost = False
        self.use_cross_entropy_cost = False
        self.use_merged_softmax_cross_entropy_cost = True

        # Initialize Activation Function Settings
        # DEFAULT: ReLU for hidden layers and Sigmoid output layer
        self.use_leaky_relu = [True] * (len(self.w) - 1) + [False]
        self.use_softmax = [False] * (len(self.w) - 1) + [True]
        self.use_sigmoid = [False] * (len(self.w))
        self.use_relu = [False] * (len(self.w))
        self.use_linear = [False] * len(self.w)
        self.use_tanh = [False] * len(self.w)


    # Function to perform NN training steps (iterative prediction / backpropagation)
    def train_network(self, data, labels, iter):

        # Reshape vector into 2D array if necessary
        if labels.ndim == 1:
            labels.shape = (1, -1)

        # Loop over random batches of data for "iter" training iterations
        for ation in np.arange(iter):
            if ation%1000 == 0: print("Current Training Step: " + str(ation))
            batch_idx = np.random.choice(data.shape[1], size=self.training_batch_size, replace=False)
            X = data[:,batch_idx]
            Y = labels[:,batch_idx]
            self.predict(X)
            self.learn(Y)

    # Perform a single prediction-only step over a given dataset
    def classify_data(self, data):
        self.predict(data)
        return self.y_hat

    # Function to perform NN prediction on a matrix of data columns
    def predict(self, X):

        # Empty stored training values
        self.a = [X]
        self.z = []
        
        # Loop over all layers
        for i in np.arange(len(self.layersizes) - 1):

            # Calculate Z (pre-activated node values for layer)
            z = np.matmul(self.w[i], self.a[i]) + np.broadcast_to(self.b[i], (self.training_batch_size, self.b[i].shape[0])).transpose()
            self.z.append(z)

            # Calculate A (activated node values for layer)
            if self.use_leaky_relu[i]: a = leaky_ReLU(z, self.leaky_relu_rates[i])
            elif self.use_relu[i]: a = ReLU(z)
            elif self.use_linear[i]: a = linear(z)
            elif self.use_tanh[i]: a = tanh(z)
            elif self.use_sigmoid[i]: a = sigmoid(z)
            elif self.use_softmax[i]: a = softmax(z)
            else: a = sigmoid(z)
            self.a.append(a)

        # Store prediction
        self.y_hat = self.a[len(self.a) - 1]

    # Function to perform backpropagation on network weights after a prediction has been stored in self.y_hat
    def learn(self, Y):

        # Reshape vector into 2D array if necessary
        if Y.ndim == 1: Y.shape = (1, -1)

        # Store number of datapoints, create tracking variables
        m = Y.shape[1]
        
        # Calculate Outer Layer Loss Function Derivatiove dL/dA
        if self.use_huber_cost: dL = d_huber(Y, self.y_hat, self.huber_cost_delta)
        elif self.use_hellinger_cost: dL = d_hellinger(Y, self.y_hat)
        elif self.use_quadratic_cost: dL = d_quadratic(Y, self.y_hat)
        elif self.use_cross_entropy_cost: dL = d_cross_entropy(Y, self.y_hat)
        elif self.use_merged_softmax_cross_entropy_cost: dL = d_merged_softmax_cross_entropy(Y, self.y_hat)
        else: dL = d_hellinger(Y, self.y_hat)
        
        # Loop over layers backwards
        for i in np.flip(np.arange(len(self.w))):
            
            # Calculate Activation Function Derivative dA/dZ
            if self.use_leaky_relu[i]: dA = d_leaky_ReLU(self.z[i], self.leaky_relu_rates[i])
            elif self.use_relu[i]: dA = d_ReLU(self.z[i])
            elif self.use_linear[i]: dA = d_linear(self.z[i])
            elif self.use_tanh[i]: dA = d_tanh(self.z[i])
            elif self.use_sigmoid[i]: dA = d_sigmoid(self.z[i])
            elif self.use_softmax[i]:
                if self.use_merged_softmax_cross_entropy_cost: dA = 1 # softmax derivative combined with loss derivative
                else: dA = d_softmax(self.z[i]) # returns softmax derivative matrix for special case handled below
            else: dA = d_sigmoid(self.z[i])

            # Calculated pre-activated node derivative dL/dZ
            # Special Case handling for Softmax Error matrix (de-vectorized approach)
            if i == (len(self.w) - 1):
                if self.use_softmax[i] and not self.use_merged_softmax_cross_entropy_cost:
                    dz = []
                    for j in np.arange(len(dA)):
                        dz.append(np.matmul(dL[:,j].reshape(1,-1), dA[j]))
                    dz = np.array(dz).reshape(-1, m)
                else:
                    dz = dL * dA
                
                prev_dz = dz

            else:
                ness = np.matmul(self.w[i + 1].T, prev_dz)
                if self.use_softmax[i] and not self.use_merged_softmax_cross_entropy_cost:
                    dz = []
                    for j in np.arange(len(dA)):
                        dz.append(np.matmul(ness[:, j].reshape(1, -1), dA[j]))
                    dz = np.array(dz).reshape(-1, m)
                else:
                    dz = ness * dA
                    
                prev_dz = dz

            # Calculate Weight Derivatives (dL/dW and dL/dB) and Apply Learning Functions
            dw = (1/m) * np.matmul(dz, self.a[i].T)
            db = ((1/m) * np.sum(dz, axis=1, keepdims=True)).reshape((self.b[i].shape[0],))
            self.w[i] = self.w[i] - self.learning_rates[i] * dw
            self.b[i] = self.b[i] - self.learning_rates[i] * db

'''
COST FUNCTION DERIVATIVES
'''

def d_huber(Y, Y_hat, delta):
    if np.linalg.norm(Y_hat - Y) < delta: return Y_hat - Y
    else: return delta * np.sign(Y_hat - Y)

def d_hellinger(Y, Y_hat):
    return (1/np.sqrt(2))*(np.ones(Y.shape) - np.divide(np.sqrt(Y), np.sqrt(Y_hat)))

def d_quadratic(Y, Y_hat):
    return Y_hat - Y

def d_cross_entropy(Y, Y_hat):
    if 0 in (Y_hat * (np.ones(Y_hat.shape) - Y_hat)):
        return np.zeros(Y_hat.shape)
    else:
        return np.divide((Y_hat - Y), (Y_hat * (np.ones(Y_hat.shape) - Y_hat)))

def d_merged_softmax_cross_entropy(Y, Y_hat):
    return Y_hat - Y

''' 
ACTIVATION FUCNTIONS 
'''
def leaky_ReLU(x, e):
    return np.maximum(e*x, x)

def ReLU(x):
    return np.maximum(0, x)

def linear(x):
    return x

def tanh(x):
    return (np.exp(x) - np.exp(-1*x))/(np.exp(x) + np.exp(-1*x))

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def softmax(x):
    safe_x = np.exp(x - np.max(x))
    return safe_x / safe_x.sum()

''' 
ACTIVATION FUCNTION DERIVATIVES 
'''

def d_leaky_ReLU(x, e):
    return np.where(x > 0, 1.0, e)

def d_ReLU(x):
    return np.where(x > 0, 1.0, 0)

def d_linear(x):
    return 1

def d_tanh(x):
    return 1 - (tanh(x))**2

def d_sigmoid(x):
    return sigmoid(x)*(1 - sigmoid(x))

def d_softmax(x):
    derivs = []
    smax = softmax(x)
    for i in np.arange(smax.shape[1]):
        derivs.append(np.diag(smax[:,i]) - smax[:,i] * smax[:,i].reshape(-1, 1) )

    return derivs