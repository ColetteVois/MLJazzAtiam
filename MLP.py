from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig('fig.png')

class MLP(nn.Module):
    def __init__(self, alpha_size):
        super(MLP, self).__init__()
        self.alpha_size = alpha_size
        self.lin1 = nn.Linear(8*alpha_size, 100)
        self.lin2 = nn.Linear(100, 50)
        self.lin3 = nn.Linear(50, 100)
        self.lin4 = nn.Linear(100,8*alpha_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.dropout = nn.Dropout(p=0)

        # self.layers = nn.Sequential(
        #     nn.Linear(8*alpha_size, 100),
        #     nn.ReLU(),
        #     nn.Linear(100, 50),
        #     nn.ReLU(),
        #     nn.Linear(50, 100),
        #     nn.ReLU(),
        #     nn.Linear(100,8*alpha_size)
        # )
    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        output = x.view(x.size(0), -1)
        output = F.relu(self.lin1(output))
        output = F.relu(self.lin2(output))
        output = F.relu(self.lin3(output))
        output = self.lin4(output)
        output = output.view(output.size(0), 8, self.alpha_size)
        output = self.softmax(output)
        return output


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, model, optimizer, criterion):
    batch_size = input_tensor.size()[0]

    optimizer.zero_grad()

    loss = 0

    output_tensor = model(input_tensor)

    loss += criterion(output_tensor, target_tensor)

    loss.backward()

    optimizer.step()

    return loss.item()/batch_size

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent+0.000001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(model, data_loader, device, print_every=1000, plot_every=500, learning_rate=0.0001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    loss_function = nn.MSELoss().to(device)

    for iter, batch in enumerate(data_loader):
        input_tensor = batch[:,:8].to(torch.float).to(device)#, requires_grad=True)
        target_tensor = batch[:,8:].to(torch.float).to(device)

        loss = train(input_tensor, target_tensor, model, optimizer, loss_function)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / len(data_loader)),
                                         iter, iter / len(data_loader) * 100, print_loss_avg))

        if iter % plot_every == 1:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    showPlot(plot_losses)
