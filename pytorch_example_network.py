import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.datasets import load_iris
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

# Establish Data
XT, YT = load_iris(return_X_y = True)
newYT = np.zeros([len(YT), 3])
for i, y in enumerate(YT):
    newYT[i][y] = 1
X_train, X_test, Y_train, Y_test = train_test_split(XT, newYT, test_size=0.2)

# Establish Params
epochs = 1000
net = Net()
dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
loader = DataLoader(dataset, batch_size=32, shuffle=True)
ll = nn.HuberLoss(delta=1.0)
oo = optim.Adam(net.parameters(), lr=1e-4)

# Train the Neural Network
for epoch in range(epochs):

    running_loss = 0.0
    for i, data in enumerate(loader, 0):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        oo.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = ll(outputs, labels)
        loss.backward()
        oo.step()

        # print statistics
        running_loss += loss.item()

    if epoch % 100 == 99:    # print every 100 epochs
        print(f'[{epoch + 1}] loss: {running_loss / 2000:.3f}')
        running_loss = 0.0

print('Finished Training')


X = torch.Tensor(X_test)
Y = torch.Tensor(Y_test)
while True:
    i = np.random.randint(0, X.shape[0], 1)
    Y_hat = net.forward(X[i, :])
    print(Y_hat)
    print(Y[i, :])
    input()