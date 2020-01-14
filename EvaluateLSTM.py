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

def eval(input_tensor, target_tensor, encoder, decoder, it, print_value = False):
    encoder_hidden = encoder.initHidden(1)

    errors = 0
    total = 0

    input_length = input_tensor.size()[1]

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[:,ei], encoder_hidden)

    decoder_hidden = encoder_hidden

    #decoder_input = torch.autograd.Variable(torch.zeros(1, 1)).type(torch.LongTensor).to(device)
    decoder_input = (25*torch.autograd.Variable(torch.zeros(1, 1))).type(torch.LongTensor)

    output_tensor = []

    error = False

    # for di in range(input_length):
    #      decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    #      #decoder_guess = decoder_output.max(2)[1]
    #      _, decoder_guess = decoder_output.topk(1)
    #      output_tensor.append(decoder_guess)
    #      if decoder_guess != target_tensor[:,di]:
    #          errors += 1
    #          error = True
    #      #decoder_input = target_tensor[:,di].view([1,1])
    #      decoder_input = decoder_guess.squeeze().detach().view([1,1])
    #      total += 1

    use_teacher_forcing = False# if random.random() < 0.5 else False

    if use_teacher_forcing:
        for di in range(input_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            output_tensor.append(topi)
            decoder_input = target_tensor[:,di].view([1,1])
    else:
        for di in range(input_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            output_tensor.append(topi)
            if topi != target_tensor[:,di]:
                errors += 1
                error = True
            decoder_input = topi.squeeze().detach().view([1,1]) # detach from history as input
            total += 1


    if it < 10:
    #if error:
        print('input : ', input_tensor.view(-1))
        print('output : ', torch.tensor(output_tensor).view(-1))
        print('target : ', target_tensor.view(-1))

    # use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    #
    # if use_teacher_forcing:
    #     # Teacher forcing: Feed the target as the next input
    #     decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    #     #decoder_output = torch.reshape(decoder_output, (4,8))
    #     #target_tensor = torch.reshape(target_tensor,(1,4*8))
    #     for batch in range(batch_size):
    #         loss += criterion(decoder_output[batch], target_tensor[batch].type(torch.LongTensor))
    #     decoder_input = target_tensor  # Teacher forcing
    #
    # else:
    #     # Without teacher forcing: use its own predictions as the next input
    #     decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    #     #decoder_output = torch.reshape(decoder_output, (4,8))
    #     decoder_input = decoder_output.detach()  # detach from history as input
    #     for batch in range(batch_size):
    #         loss += criterion(decoder_output[batch], target_tensor[batch].type(torch.LongTensor))

    return errors, total


def evalIters(encoder, decoder, data_loader):

    total_errors = 0
    total = 0

    for iter, batch in enumerate(data_loader):
        #input_tensor = batch[:,:8,:].to(torch.int64)
        #target_tensor = batch[:,8:,:].to(torch.int64)
        input_tensor = batch[:,:8].to(torch.int64)#, requires_grad=True)
        target_tensor = batch[:,8:].to(torch.int64)

        errors, n = eval(input_tensor, target_tensor, encoder, decoder, iter, print_value = True)

        total_errors += errors
        total += n
        if iter % 5000 == 0:
            print(iter)

    return total_errors, total
