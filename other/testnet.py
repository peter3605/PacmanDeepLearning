import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(torch.nn.Module):

    def __init__(self, input_dim, fc_dim, output_dim, max_action):
        super(Actor, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        
        self.fc3 = nn.Linear(7040, fc_dim)
        self.fc4 = nn.Linear(fc_dim, output_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.max_action = max_action


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.fc3(x))
        x = self.tanh(self.fc4(x))
        
        return x


class Critic(nn.Module):

    def __init__(self, input_dim, fc_dim, output_dim, max_action):
        super(Critic, self).__init__()

        # Critic 1
        self.c1_conv1 = nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=0)
        self.c1_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        
        self.c1_fc3 = nn.Linear(7040, fc_dim)
        self.c1_fc4 = nn.Linear(fc_dim, output_dim)

        # Critic 2
        self.c2_conv1 = nn.Conv2d(input_dim, 16, kernel_size=3, stride=1, padding=0)
        self.c2_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        
        self.c2_fc3 = nn.Linear(7040, fc_dim)
        self.c2_fc4 = nn.Linear(fc_dim, output_dim)


    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        
        # Critic 1
        x1 = self.relu(self.c1_conv1(xu))
        x1 = self.relu(self.c1_conv2(x1))
        x1 = self.relu(self.c1_fc3(x1))
        x1 = self.tanh(self.c1_fc4(x1))

        # Critic 2
        x2 = self.relu(self.c2_conv1(xu))
        x2 = self.relu(self.c2_conv2(x2))
        x2 = self.relu(self.c2_fc3(x2))
        x2 = self.tanh(self.c2_fc4(x2))

        return x1, x2


    def getQ(self, x, u):
        xu = torch.cat([x, u], 1)

        x = self.relu(self.c1_conv1(xu))
        x = self.relu(self.c1_conv2(x))
        x = self.relu(self.c1_fc3(x))
        x = self.tanh(self.c1_fc4(x))

        return x

