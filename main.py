import create_dataloader as dl
import EncoderLSTM as lstm
import Evaluate as eval
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


batch_size = 50
alphabet = 'a0'
listeA0 = ['N','A:maj', 'A#:maj','B:maj','C:maj', 'C#:maj','D:maj','D#:maj','E:maj','F:maj', 'F#:maj','G:maj','G#:maj', 'A:min', 'A#:min','B:min','C:min', 'C#:min','D:min','D#:min','E:min','F:min', 'F#:min','G:min','G#:min']
chordSeqDatasetTrain = dl.ChordSequencesDatasetClass('../data/preprocessed_data_train.csv', transform=transforms.Compose([dl.ReduChord(alphabet), dl.oneHotVector(listeA0)]))
print(len(chordSeqDatasetTrain))
dataloader_train = DataLoader(chordSeqDatasetTrain, batch_size=batch_size, shuffle=True, num_workers=4)


chordSeqDatasetTest = dl.ChordSequencesDatasetClass('../data/preprocessed_data_test.csv', transform=transforms.Compose([dl.ReduChord(alphabet), dl.oneHotVector(listeA0)]))
print(len(chordSeqDatasetTest))
dataloader_test = DataLoader(chordSeqDatasetTest, batch_size=1, shuffle=True, num_workers=4)

print('Dataloader created\n')

hidden_size = 256

alpha_size = 25
#data_size = batch_size*alpha_size*8
data_size = batch_size*8*hidden_size

encoder = lstm.EncoderRNN(alpha_size, hidden_size, alpha_size).to(device)
decoder = lstm.DecoderRNN(hidden_size, alpha_size, alpha_size).to(device)

lstm.trainIters(encoder, decoder, dataloader_train, print_every=500, plot_every = 2, learning_rate = 0.0001)

torch.save(encoder.state_dict(), 'encoder.dict')
torch.save(decoder.state_dict(), 'decoder.dict')

#Décommenter pour test

# errors,total = eval.evalIters(encoder, decoder, dataloader_test)
# print(errors/total*100, '% d erreurs')





# encoder.load_state_dict(torch.load('encoder.dict'))
# decoder.load_state_dict(torch.load('decoder.dict'))
