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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dict_size, batch_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dict_size = dict_size
        self.batch_size = batch_size
        
        self.embedding = nn.Embedding(dict_size, hidden_size)
        self.lstm = nn.LSTM(input_size=input_size, num_layers=1, hidden_size=hidden_size, batch_first = True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def initHidden(self):
        hidden = torch.zeros(1, self.batch_size, self.hidden_size, device=device)
        return (hidden, hidden)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dict_size, batch_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.embedding = nn.Embedding(dict_size, hidden_size)
        #self.lstm = nn.LSTM(output_size, hidden_size, batch_first = True)
        self.lstm = nn.LSTM(input_size=output_size, num_layers=1, hidden_size=hidden_size, batch_first = True)
        self.out = nn.Linear(hidden_size,1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        print('input', input.size())
        output = self.embedding(input)
        print('embeded', output)
        print('hidden', hidden[0])
        output = F.relu(output)
        print('relu', output)
        output, hidden = self.lstm(output, hidden)
        print('camarche lstm', output, hidden[0])
        print('self.out(output[0])', self.out(output))
        output = self.softmax(self.out(output))
        print('softmax', output)
        return output, hidden

    def initHidden(self):
        hidden = torch.zeros(1, self.batch_size, self.hidden_size, device=device)
        return (hidden, hidden)


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)
    decoder_input = torch.zeros([4,8], device=device).to(torch.int64)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        print('decoder_output',decoder_output)
        decoder_output = torch.reshape(decoder_output, (4,8))
        print('loss',decoder_output.size(), target_tensor.size() )
        loss += criterion(decoder_output, target_tensor)
        decoder_input = target_tensor  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        print('decoder_output',decoder_output)
        decoder_output = torch.reshape(decoder_output, (4,8))
        decoder_input = decoder_output.detach()  # detach from history as input
        loss += criterion(decoder_output, target_tensor)

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(encoder, decoder, data_loader, n_iter, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()

    i = 0
    for batch in data_loader:
        if i <= n_iter:
            #input_tensor = batch[:,:8,:].to(torch.int64)
            #target_tensor = batch[:,8:,:].to(torch.int64)
            input_tensor = batch[:,:8].to(torch.int64)
            target_tensor = batch[:,8:].to(torch.int64)
            print()

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, loss_function)

            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
        i = i+1
    showPlot(plot_losses)
