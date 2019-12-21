import create_dataloader as dl
import EncoderLSTM as lstm
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from chordUtil import reduChord
from sklearn import preprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alphabet = 'a0'
listeA0 = ['N','A:maj', 'A#:maj','B:maj','C:maj', 'C#:maj','D:maj','D#:maj','E:maj','F:maj', 'F#:maj','G:maj','G#:maj', 'A:min', 'A#:min','B:min','C:min', 'C#:min','D:min','D#:min','E:min','F:min', 'F#:min','G:min','G#:min']
chordSeqDatasetTrain = dl.ChordSequencesDatasetClass('data/preprocessed_data_test.csv', transform=transforms.Compose([dl.ReduChord(alphabet), dl.oneHotVector(listeA0)]))
#print(chordSeqDatasetTrain[1])
dataloader = DataLoader(chordSeqDatasetTrain, batch_size=4, shuffle=True, num_workers=4)

print('Dataloader created\n')

hidden_size = 256
batch_size = 4
alpha_size = 25
#data_size = batch_size*alpha_size*8
data_size = batch_size*8*hidden_size
n_iter = 100000

encoder = lstm.EncoderRNN(hidden_size, hidden_size, alpha_size, batch_size).to(device)
decoder = lstm.DecoderRNN(hidden_size, hidden_size, alpha_size, batch_size).to(device)

lstm.trainIters(encoder, decoder, dataloader, n_iter, print_every=500)
