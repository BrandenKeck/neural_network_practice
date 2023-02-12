import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

class Net(nn.Module):

    def __init__(self, s, lt):
        super(Net, self).__init__()
        fcs = (len(s)-1)*[None]
        for idx in range(1, len(s)): 
            fcs[idx-1]=nn.Linear(s[idx-1], s[idx])
        self.layers = nn.ModuleList(fcs)
        self.layertypes = lt

    def forward(self, x):
        for idx in range(len(self.layers)):
            if self.layertypes[idx] == "relu":
                x = F.relu(self.layers[idx](x))
            elif self.layertypes[idx] == "softmax":
                x = F.softmax(self.layers[idx](x), dim=1)
        return x

class neural_network():

    # Init Network Class
    def __init__(self, structure=[4, 128, 64, 3], 
                        layertypes = ["relu", "relu", "softmax"],
                        learning_rate=1e-4, huber_delta = 1.0, 
                        batch_size=32, epochs=1000):
        self.net = Net(structure, layertypes)
        self.lr = learning_rate
        self.delta = huber_delta
        self.batch_size = batch_size
        self.epochs = epochs

    # Train Function
    def train_network(self, X, Y):
        
        # Establish Params
        dataset = TensorDataset(torch.Tensor(X), torch.Tensor(Y))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        ll = nn.HuberLoss(delta=self.delta)
        oo = optim.Adam(self.net.parameters(), lr=1e-4)

        # Train the Neural Network
        for epoch in range(self.epochs):

            running_loss = 0.0
            for i, data in enumerate(loader, 0):
                
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                oo.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = ll(outputs, labels)
                loss.backward()
                oo.step()

                # print statistics
                running_loss += loss.item()

            if epoch % 100 == 99:    # print every 100 epochs
                print(f'[{epoch + 1}] loss: {running_loss / i}')
                running_loss = 0.0

        print('Finished Training')

    # Predict from network
    def predict_network(self, X):
        X = torch.Tensor(X)
        return self.net.forward(X)
