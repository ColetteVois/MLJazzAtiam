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

# def showPlot(points):
#     plt.figure()
#     fig, ax = plt.subplots()
#     # this locator puts ticks at regular intervals
#     loc = ticker.MultipleLocator(base=0.2)
#     ax.yaxis.set_major_locator(loc)
#     plt.plot(points)
#     plt.savefig('fig.png')

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dict_size, dropout, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dict_size = dict_size
        self.device = device

        self.embedding = nn.Embedding(dict_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, num_layers=1, hidden_size=hidden_size, batch_first = True, dropout = dropout)

    def forward(self, input, hidden):
        batch_size = input.size()[0]
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded.view([batch_size, 1,self.hidden_size]), hidden)
        return output, hidden

    def initHidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.hidden_size, device=self.device)
        return (hidden, hidden)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dict_size, dropout, device):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(dict_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, num_layers=1, hidden_size=hidden_size, batch_first = True, dropout = dropout)
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
    '''
        Fonction effectuant un passage à travers le systeme encodeur-decodeur et mettant à jour les paramètres.
        inputs :
        - input_tensor : torch.tensor de taille [batch_size, 8]
        - target_tensor : torch.tensor de taille [batch_size, 8]
        - encoder : nn.Module
        - decoder : nn.Module
        - encoder_optimizer : torch optimizer
        - decoder_optimizer : torch optimizer
        - criterion : fonction d'erreur pytorch
        - teacher_forcing_ratio : float entre 0 et 1 : probabilité d'utiliser teacher forcing
        - device : torch device
    '''

    batch_size = input_tensor.size()[0]
    print(encoder)

    encoder_hidden = encoder.initHidden(batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0

    input_length = input_tensor.size()[1]

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[:,ei], encoder_hidden)

    decoder_hidden = encoder_hidden

    decoder_input = torch.autograd.Variable(torch.zeros(batch_size, 1)).type(torch.LongTensor).to(device)

    #Si teacher forcing : on utilise le ground truth en entrée du decoder
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



def trainIters(encoder, decoder, data_loader, device, epochRatio, learning_rate=0.0001):
    '''
        Boucle d'entraînement de l'encodeur-decodeur
        inputs :
        - encoder : nn.Module
        - decoder : nn.Module
        - data_loader : pandas dataloader contenant les données d'entrainement
        - device : torch Device
        - epochRatio : float entre 0 et 1 : permet de calculer la probabilité de teacher forcing
        - learning_rate : float

        output :
        - plot_losses : liste contenant la valeur de l'erreur à chaque iteration
    '''
    plot_losses = []

    encoder_optimizer = optim.Adam(encoder.parameters(), lr = learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr = learning_rate)
    loss_function = nn.NLLLoss().to(device)

    for iter, batch in enumerate(data_loader):
        input_tensor = batch[:,:8].to(torch.int64).to(device)
        target_tensor = batch[:,8:].to(torch.int64).to(device)

        tf_ratio = 1 - epochRatio

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, loss_function, tf_ratio, device)

        plot_losses.append(loss)
    return plot_losses
