import torch.nn as nn

class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)  # Added second hidden layer
        self.fc3 = nn.Linear(64, 9)    # Added output layer

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))  # Using ReLU activation for hidden layers
        x = self.fc3(x)
        return x