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
from create_dataloader import chordFromIndex
import get_the_list


def eval(input_tensor, target_tensor, encoder, decoder, it, print_value = False):
    '''
        Fonction evaluant un batch
        inputs :
        - input_tensor : torch.tensor de taille [batch_size, 8]
        - target_tensor : torch.tensor de taille [batch_size, 8]
        - encoder : nn.Module
        - decoder : nn.Module
        - it : int
        - print_value : booléen. True si on veut afficher des exemples

        outputs :
        - errors : int
        - total : int
    '''

    batch_size = input_tensor.size()[0]

    encoder_hidden = encoder.initHidden(batch_size)

    errors = 0
    total = 0

    input_length = input_tensor.size()[1]

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[:,ei], encoder_hidden)

    decoder_hidden = encoder_hidden

    decoder_input = (25*torch.autograd.Variable(torch.zeros(batch_size, 1))).type(torch.LongTensor)

    output_tensor = []

    for di in range(input_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        output_tensor.append(topi)
        if topi != target_tensor[:,di]:
            errors += 1
        decoder_input = topi.squeeze().detach().view([1,1]) # detach from history as input
        total += 1


    if print_value:
        if it < 10:
            print('input : ', input_tensor.view(-1))
            #print(chordFromIndex(input_tensor.view(-1), liste))
            print('output : ', torch.tensor(output_tensor).view(-1))
            #print(chordFromIndex(torch.tensor(output_tensor).view(-1), liste))
            print('target : ', target_tensor.view(-1))
            #print(chordFromIndex(target_tensor.view(-1), liste))

    return errors, total


def evalIters(encoder, decoder, data_loader):
    '''
    Boucle d'evaluation de l'encodeur-decodeur
    inputs :
    - encoder : nn.Module
    - decoder : nn.Module
    - data_loader : pandas dataloader contenant les données de test

    outputs :
    - total_errors : int
    - total : int
    '''

    total_errors = 0 #nombre d'erreurs effectuées
    total = 0 #nombre total d'accord comparés

    for iter, batch in enumerate(data_loader):
        input_tensor = batch[:,:8].to(torch.int64)
        target_tensor = batch[:,8:].to(torch.int64)

        errors, n = eval(input_tensor, target_tensor, encoder, decoder, iter, print_value = True)

        total_errors += errors
        total += n

    return total_errors, total
