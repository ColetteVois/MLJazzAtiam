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


listeA0 = ['N','A:maj', 'A#:maj','B:maj','C:maj', 'C#:maj','D:maj','D#:maj','E:maj','F:maj', 'F#:maj','G:maj','G#:maj', 'A:min', 'A#:min','B:min','C:min', 'C#:min','D:min','D#:min','E:min','F:min', 'F#:min','G:min','G#:min']

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

        return chords
        #return chords.clone().detach()


class ReduChord(object):
    def __init__(self, alpha = 'a0'):
        self.alpha = alpha #Alphabet used for reduction

    def __call__(self, chords, alpha = 'a0'):
        chords_redu = []
        for j in range(16):
            chords_redu.append(reduChord(chords[j], self.alpha))
        return pd.Series(chords_redu)

class ClassVector(object):
    def __init__(self, liste = listeA0):
        self.liste = liste #Alphabet used for reduction

    def __call__(self, chords, liste = listeA0):
        vect = torch.zeros([16], dtype=torch.int)
        for i in range(16):
            numChord = self.liste.index(chords[i])
            vect[i] = numChord
        return vect

class OneHotVector(object):
    def __init__(self, liste = listeA0):
        self.liste = liste #Alphabet used for reduction

    def __call__(self, chords, liste = listeA0):
        oneVect = torch.zeros([16, 25], dtype=torch.int64)
        for i in range(16):
            numChord = liste.index(chords[i])
            oneVect[i, numChord] = 1
        return oneVect

def chordFromIndex(idx_vect, liste):
    chord_seq = "";
    for idx in idx_vect:
        chord_seq = chord_seq + liste[idx] + ", "
    return chord_seq
