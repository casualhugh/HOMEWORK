# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
input_dim = 784
output_dim = 10


class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        outputs = self.main(x)
        return outputs

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 130),
            nn.Tanh(),
            nn.Linear(130, 10),
            nn.LogSoftmax(dim=1)
        )
    def forward(self, x):
        outputs = self.main(x)
        return outputs

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4)
        self.fc1 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = self.fc1(x)
        return F.log_softmax(x)
