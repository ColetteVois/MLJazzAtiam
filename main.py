import create_dataloader as dl
import EncoderLSTM as lstm
import MLP
import EvaluateLSTM as evalLSTM
import EvaluateMLP as evalMLP
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from chordUtil import reduChord
from sklearn import preprocessing
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = 'LSTM', help = 'Model type (MLP or LSTM)')
parser.add_argument('--device', type = str, default = 'cpu', help = 'Device to use')
args = parser.parse_args()
print(args)


if (args.device != 'cpu'):
    # Enable CuDNN optimization
    torch.backends.cudnn.benchmark=True

args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print('Optimization will be on ' + str(args.device) + '.')
dtype = torch.cuda.FloatTensor

batch_size = 50
alphabet = 'a0'
listeA0 = ['N','A:maj', 'A#:maj','B:maj','C:maj', 'C#:maj','D:maj','D#:maj','E:maj','F:maj', 'F#:maj','G:maj','G#:maj', 'A:min', 'A#:min','B:min','C:min', 'C#:min','D:min','D#:min','E:min','F:min', 'F#:min','G:min','G#:min']
alpha_size = 25
hidden_size = 256

if args.model == "MLP":
    #Dataloader Train
    chordSeqDatasetTrain = dl.ChordSequencesDatasetClass('../data/preprocessed_data_train.csv', transform=transforms.Compose([dl.ReduChord(alphabet), dl.OneHotVector(listeA0)]))
    print("Number of training sequences : ", len(chordSeqDatasetTrain))
    dataloader_train = DataLoader(chordSeqDatasetTrain, batch_size=batch_size, shuffle=True, num_workers=4)

    #Dataloader Test
    chordSeqDatasetTest = dl.ChordSequencesDatasetClass('../data/preprocessed_data_test.csv', transform=transforms.Compose([dl.ReduChord(alphabet), dl.OneHotVector(listeA0)]))
    print("Number of test sequences : ", len(chordSeqDatasetTest))
    dataloader_test = DataLoader(chordSeqDatasetTest, batch_size=1, shuffle=True, num_workers=4)

    print('Dataloader created\n')

    #Model
    modelMLP = MLP.MLP(alpha_size).to(args.device)

    #Start Training
    print("Start Training")
    MLP.trainIters(modelMLP, dataloader_train)
    torch.save(modelMLP.state_dict(), 'MLP.dict')

    #Start Test
    print("Start Test")
    # modelMLP.load_state_dict(torch.load('MLP.dict'))
    errors,total = evalMLP.evalIters(modelMLP, dataloader_test)

if args.model == "LSTM":
    #Dataloader Train
    chordSeqDatasetTrain = dl.ChordSequencesDatasetClass('../data/preprocessed_data_train.csv', transform=transforms.Compose([dl.ReduChord(alphabet), dl.ClassVector(listeA0)]))
    print("Number of training sequences : ", len(chordSeqDatasetTrain))
    dataloader_train = DataLoader(chordSeqDatasetTrain, batch_size=batch_size, shuffle=True, num_workers=4)

    #Dataloader Test
    chordSeqDatasetTest = dl.ChordSequencesDatasetClass('../data/preprocessed_data_test.csv', transform=transforms.Compose([dl.ReduChord(alphabet), dl.ClassVector(listeA0)]))
    print("Number of test sequences : ", len(chordSeqDatasetTest))
    dataloader_test = DataLoader(chordSeqDatasetTest, batch_size=1, shuffle=True, num_workers=4)

    print('Dataloader created\n')

    encoder = lstm.EncoderRNN(alpha_size, hidden_size, alpha_size).to(args.device)
    decoder = lstm.DecoderRNN(hidden_size, alpha_size, alpha_size).to(args.device)

    #Start Training
    print("Start Training")
    lstm.trainIters(encoder, decoder, dataloader_train, args.device, print_every=500, plot_every = 2, learning_rate = 0.0001)
    torch.save(encoder.state_dict(), 'encoder.dict')
    torch.save(decoder.state_dict(), 'decoder.dict')

    #Start Test
    print("Start Test")
    # encoder.load_state_dict(torch.load('encoder_save.dict'))
    # decoder.load_state_dict(torch.load('decoder_save.dict'))
    #errors,total = evalLSTM.evalIters(encoder, decoder, dataloader_test)


#print(errors/total*100, '% d erreurs')
