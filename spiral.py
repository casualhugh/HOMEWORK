# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# HID of 6 seems 50/50 7 is prefered
class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        self.inp = nn.Linear(2, num_hid)
        self.out = nn.Linear(num_hid, 1)
        self.hid1 = 0

    def forward(self, input):
        x = input[...,0]
        y = input[...,1]
        r = torch.sqrt(x**2+y**2)
        a = torch.atan2(y,x)
        converted = torch.stack([r,a],1)
        f_input = torch.tanh(self.inp(converted))
        self.hid1 = torch.tanh(f_input)
        f_output = torch.sigmoid(self.out(self.hid1))
        return f_output

# 8 or 9 seesms to work
class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        self.inp = nn.Linear(2, num_hid)
        self.hidden = nn.Linear(num_hid, num_hid)
        self.outlin = nn.Linear(num_hid, 1)
        self.hid1 = 0
        self.hid2 = 0


    def forward(self, input):
        f_input = self.inp(input)
        self.hid1 = torch.tanh(f_input)
        f_hidden = self.hidden(self.hid1)
        self.hid2 = torch.tanh(f_hidden)
        output = torch.sigmoid(self.outlin(self.hid2))
        return output

def graph_hidden(net, layer, node):
    xrange = torch.arange(start=-7, end=7.1, step=0.01, dtype=torch.float32)
    yrange = torch.arange(start=-6.6, end=6.7, step=0.01, dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1), ycoord.unsqueeze(1)), 1)

    with torch.no_grad():  # suppress updating of gradients
        if layer <= 2:
            hidden = net.inp(grid)
        else:
            hidden = net.hidden(grid)
        hidden = torch.tanh(hidden)
        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange, yrange, hidden[:,node].cpu().view(yrange.size()[0], xrange.size()[0]), shading='auto',
                       cmap='Wistia')
