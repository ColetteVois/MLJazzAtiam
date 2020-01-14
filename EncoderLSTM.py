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

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dict_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dict_size = dict_size
        self.device = device

        self.embedding = nn.Embedding(dict_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, num_layers=1, hidden_size=hidden_size, batch_first = True)

    def forward(self, input, hidden):
        batch_size = input.size()[0]
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded.view([batch_size, 1,self.hidden_size]), hidden)
        return output, hidden

    def initHidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        return (hidden, hidden)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dict_size, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(dict_size, hidden_size)
        #self.lstm = nn.LSTM(output_size, hidden_size, batch_first = True)
        self.lstm = nn.LSTM(input_size=hidden_size, num_layers=1, hidden_size=hidden_size, batch_first = True)
        self.out = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        #input = input.type(torch.FloatTensor)
        output = F.relu(embedded)
        output, hidden = self.lstm(embedded, hidden)
        output = self.softmax(self.out(output))
        return output, hidden

    def initHidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        return (hidden, hidden)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio, device):
    batch_size = input_tensor.size()[0]

    encoder_hidden = encoder.initHidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    input_length = input_tensor.size()[1]

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[:,ei], encoder_hidden)

    decoder_hidden = encoder_hidden

    decoder_input = torch.autograd.Variable(torch.zeros(batch_size, 1)).type(torch.LongTensor).to(device)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(input_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            for batch in range(batch_size):
                loss += criterion(decoder_output[batch], target_tensor[batch,di].view([1]))
            decoder_input = target_tensor[:,di].view([batch_size,1])
    else:
        for di in range(input_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            for batch in range(batch_size):
                loss += criterion(decoder_output[batch], target_tensor[batch,di].view([1]))
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().view([batch_size,1]) # detach from history as input

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()/batch_size/8

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

def trainIters(encoder, decoder, data_loader, device, teacher_forcing_ratio, print_every=1000, plot_every=500, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
    loss_function = nn.NLLLoss().to(device)

    for iter, batch in enumerate(data_loader):
        input_tensor = batch[:,:8].to(torch.int64).to(device)#, requires_grad=True)
        target_tensor = batch[:,8:].to(torch.int64).to(device)

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, loss_function, teacher_forcing_ratio, device)

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
