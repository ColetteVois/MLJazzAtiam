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


class ChordSequencesDatasetClass(Dataset):
    def __init__(self, csv_file, transform=None):
        self.chord_seq = pd.read_csv(csv_file, sep = ";", header = None)
        self.transform = transform

    def __len__(self):
        return len(self.chord_seq)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.chord_seq.iloc[idx, :]
        sample = sample.str.split("%", expand=True)

        chords = sample.iloc[:,0]
        beat = sample.iloc[:,1]
        key = sample.iloc[0,2]

        if self.transform:
            chords = self.transform(chords)

        sample = {'chords' : chords, 'beat' : beat, 'key' : key}

        return sample


class ReduChord(object):
    def __init__(self, alpha = 'a0'):
        self.alpha = alpha #Alphabet used for reduction

    def __call__(self, chords, alpha = 'a0'):
        chords_redu = []
        for j in range(16):
            chords_redu.append(reduChord(chords[j], self.alpha))
        return pd.Series(chords_redu)

#alphabet = 'a0'
#chordSeqDatasetTrain = ChordSequencesDatasetClass('../data/preprocessed_data_train.csv', ReduChord(alphabet))
#dataloader = DataLoader(chordSeqDatasetTrain, batch_size=4, shuffle=True, num_workers=4)
