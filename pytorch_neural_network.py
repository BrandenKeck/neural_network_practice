import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class neural_network(nn.Module):

    def __init__(self, layersizes, activations, learning_rate=0.01,
                 optimizer='adam', losses='huber', huber_delta = 1.0,
                 training_epochs = 1):

        super(neural_network, self).__init__()

        # Establish a Model Layout
        self.layers = []
        for i in np.arange(1, len(layersizes)):
            self.layers.append(nn.Linear(layersizes[i-1], layersizes[i]))

        # Store Training Information
        self.activations = activations
        self.lr = learning_rate
        self.optim = optimizer
        self.lossf = losses
        self.delta = huber_delta
        self.training_epochs = training_epochs

    def forward(self, X):
        for i in np.arange(self.layers):
            if self.activations[i] == "relu": X = F.relu(self.layers[i](X))
            if self.activations[i] == "linear": X = F.linear(self.layers[i](X))
            if self.activations[i] == "softmax": X = F.softmax(self.layers[i](X))
        return X

    # Train the Network
    def train(self, X, Y):

        if self.lossf=='huber': ll = nn.HuberLoss(delta=self.delta)
        elif self.lossf=='crossentropy': ll = nn.CrossEntropyLoss()
        elif self.lossf=='mse': ll = nn.MSELoss()
        else: ll = nn.HuberLoss(delta=self.delta)

        if self.optim=='adam': oo = optim.SGD(self.parameters(), lr=self.lr)
        elif self.optim=='sgd': oo = optim.Adam(self.parameters(), lr=self.lr)
        else: oo = optim.SGD(self.parameters(), lr=self.lr)

        for t in range(self.training_epochs):
            if t%1000 == 0: print(f"Current Training Step: {t}")
            Y_pred = self(X)
            loss = ll(Y_pred, Y)
            # loss_list.append(loss.item())
            self.zero_grad()
            loss.backward()
            oo.step()
    