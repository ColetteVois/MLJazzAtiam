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

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def evalIters(model, data_loader, print_value = False):
    '''
    Boucle d'evaluation de l'encodeur-decodeur
    inputs :
    - model : nn.Module
    - data_loader : pandas dataloader contenant les données de test
    - print_value : booléen. True si on veut afficher des exemples

    outputs :
    - total_errors : int
    - total : int
    '''

    total_errors = 0
    total = 0

    for iter, batch in enumerate(data_loader):
        input_tensor = batch[:,:8].to(torch.float)#, requires_grad=True)
        target_tensor = batch[:,8:].to(torch.float)
        output_tensor = model(input_tensor)

        batch_size = input_tensor.size()[0]

        _, input_vect = torch.max(input_tensor,2)
        _, target_vect = torch.max(target_tensor,2)
        _, output_vect = torch.max(output_tensor,2)

        for l in range(batch_size):
            for i in range(8):
                if target_vect[l,i] != output_vect[l,i]:
                    total_errors += 1
                total +=1

        if print_value:
            if iter<10:
                print('input : ', input_vect[0])
                print('output : ', output_vect[0])
                print('target : ', target_vect[0])

    return total_errors, total
