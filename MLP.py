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

class MLP(nn.Module):
    def __init__(self, alpha_size, dropout):
        super(MLP, self).__init__()
        self.alpha_size = alpha_size
        self.softmax = nn.LogSoftmax(dim=2)

        self.layers = nn.Sequential(
            nn.Linear(8*alpha_size, 100),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(100,8*alpha_size)
        )
    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        output = x.view(x.size(0), -1)
        output = self.layers(output)
        output = output.view(output.size(0), 8, self.alpha_size)
        output = self.softmax(output)
        return output


def train(input_tensor, target_tensor, model, optimizer, criterion):
    '''
        Fonction effectuant un passage à travers le systeme encodeur-decodeur et mettant à jour les paramètres.
        inputs :
        - input_tensor : torch.tensor de taille [batch_size, 8]
        - target_tensor : torch.tensor de taille [batch_size, 8]
        - model : nn.Module
        - optimizer : torch optimizer
        - criterion : fonction d'erreur pytorch
        - teacher_forcing_ratio : float entre 0 et 1 : probabilité d'utiliser teacher forcing
    '''
    batch_size = input_tensor.size()[0]

    optimizer.zero_grad()

    model.zero_grad()

    loss = 0

    output_tensor = model(input_tensor)

    for batch in range(batch_size):
        loss += criterion(output_tensor[batch], target_tensor[batch])

    loss.backward()

    optimizer.step()

    return loss.item()/batch_size


def trainIters(model, data_loader, device, learning_rate=0.0001):
    '''
        Boucle d'entraînement de l'encodeur-decodeur
        inputs :
        - model : nn.Module
        - data_loader : pandas dataloader contenant les données d'entrainement
        - device : torch Device
        - learning_rate : float

        output :
        - plot_losses : liste contenant la valeur de l'erreur à chaque iteration
    '''

    plot_losses = []

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    #loss_function = nn.MSELoss().to(device)
    loss_function = nn.NLLLoss().to(device)

    for iter, batch in enumerate(data_loader):
        input_tensor = batch[:,:8].to(torch.float).to(device)
        target_tensor = batch[:,8:].to(torch.float).to(device)
        _, target_vect = torch.max(target_tensor,2)
        target_vect.to(device)

        loss = train(input_tensor, target_vect, model, optimizer, loss_function)

        plot_losses.append(loss)

    return plot_losses
