import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 6 input channels for state matrix with 4 possible actions to take
class DDQN(nn.Module):
    def __init__(self, width, height, num_inputs=6, num_actions=4):
        super(DDQN, self).__init__()

        self.conv1 = nn.Conv2d(num_inputs, 16, kernel_size=3, stride=1, padding=1) # = W x H
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # = W x H
        self.fc3 = nn.Linear(width*height*32, 256)
        self.fc4 = nn.Linear(256, num_actions)

    
    def forward(self, state):
        state = F.relu(self.conv1(state))
        state = F.relu(self.conv2(state))
        state = state.view(state.size(0), -1) # flatten for fc layer
        state = F.relu(self.fc3(state))
        qvalues = self.fc4(state)
        return qvalues
