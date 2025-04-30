import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import builtins
from configparser import ConfigParser

  
configur = ConfigParser()
configur.read(builtins.current_filename)
layer1=int(configur.get('architecture','layer1'))
layer2=int(configur.get('architecture','layer2'))

class DQNNet(nn.Module):
    def __init__(self, input_size, output_size, lr=1e-3):
        super(DQNNet, self).__init__()
        self.dense1 = nn.Linear(input_size, layer1)
        self.dense2 = nn.Linear(layer1, layer2)
        self.dense3 = nn.Linear(layer2, output_size)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x

    def saveModel(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.state_dict(), filename)

    def loadModel(self, filename, device):
        self.load_state_dict(torch.load(filename, map_location=device))


