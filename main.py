import create_dataloader as dl
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
import EncoderLSTM as lstm
import MLP
import EvaluateLSTM as evalLSTM
import EvaluateMLP as evalMLP
from get_the_list import list_of_alphabet


parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, default = 'LSTM', help = 'Model type (MLP or LSTM)')
parser.add_argument('--device', type = str, default = 'cpu', help = 'Device to use')
parser.add_argument('--alphabet', type = str, default = 'a0', help = 'Chords Alphabet')
parser.add_argument('--batch', type = int, default = 50, help = 'Batch size')
parser.add_argument('--hidden', type = int, default = 256, help = 'Hidden state size')
parser.add_argument('--dropout', type = float, default = 0, help = 'dropout_rate')
parser.add_argument('--teacherForcing', type = float, default = 0.5, help = 'Teacher forcing ratio in training')
parser.add_argument('--saveEncoder', type = str, default = 'encoder.dict', help = 'Name of encoder savefile')
parser.add_argument('--saveDecoder', type = str, default = 'decoder.dict', help = 'Name of decoder savefile')
args = parser.parse_args()
print(args)


max_epochs = 10


if (args.device != 'cpu'):
    # Enable CuDNN optimization
    torch.backends.cudnn.benchmark=True

args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print('Optimization will be on ' + str(args.device) + '.')
dtype = torch.cuda.FloatTensor


#Import de l'alphabet utilisé dans une liste
chordsList = list_of_alphabet(args.alphabet)
print(chordsList)
print(len(chordsList))
#chordsList.append('SOS_TOKEN')
alpha_size = len(chordsList)

if args.model == "MLP":
    #Dataloader Train
    chordSeqDatasetTrain = dl.ChordSequencesDatasetClass('../data/preprocessed_data_train.csv', transform=transforms.Compose([dl.ReduChord(args.alphabet), dl.OneHotVector(chordsList)]))
    print("Number of training sequences : ", len(chordSeqDatasetTrain))
    dataloader_train = DataLoader(chordSeqDatasetTrain, batch_size=args.batch, shuffle=True, num_workers=4)

    #Dataloader Test
    chordSeqDatasetTest = dl.ChordSequencesDatasetClass('../data/preprocessed_data_oneline.csv', transform=transforms.Compose([dl.ReduChord(args.alphabet), dl.OneHotVector(chordsList)]))
    print("Number of test sequences : ", len(chordSeqDatasetTest))
    dataloader_test = DataLoader(chordSeqDatasetTest, batch_size=args.batch//10, shuffle=True, num_workers=4)

    #Dataloader Valid
    chordSeqDatasetValid = dl.ChordSequencesDatasetClass('../data/preprocessed_data_oneline.csv', transform=transforms.Compose([dl.ReduChord(args.alphabet), dl.OneHotVector(chordsList)]))
    print("Number of validation sequences : ", len(chordSeqDatasetValid))
    dataloader_valid = DataLoader(chordSeqDatasetValid, batch_size=1, shuffle=True, num_workers=4)

    print('Dataloader created\n')

    #Model
    modelMLP = MLP.MLP(alpha_size, args.dropout).to(args.device)

    loss = []

    for epoch in range(max_epochs):
        print("Epoch ", epoch)
        #Start Training
        print("Start Training")
        modelMLP.train()
        lossep = MLP.trainIters(modelMLP, dataloader_train, args.device, print_every=20, plot_every = 2, learning_rate = 0.0001)

        loss = loss + lossep

        #Start Test
        print("Start Test")
        modelMLP.eval()
        errors,total = evalMLP.evalIters(modelMLP, dataloader_test)
        print(errors/total*100, '% d erreurs')

    torch.save(modelMLP.state_dict(), 'MLP.dict')


    # modelMLP.load_state_dict(torch.load('gpuresults/MLP_a5.dict', map_location = torch.device('cpu')))
    modelMLP.eval()
    errors,total = evalMLP.evalIters(modelMLP, dataloader_valid, print_value = True)
    print(errors/total*100, '% d erreurs')


if args.model == "LSTM":
    #Dataloader Train
    chordSeqDatasetTrain = dl.ChordSequencesDatasetClass('../data/preprocessed_data_train.csv', transform=transforms.Compose([dl.ReduChord(args.alphabet), dl.ClassVector(chordsList)]))
    print("Number of training sequences : ", len(chordSeqDatasetTrain))
    dataloader_train = DataLoader(chordSeqDatasetTrain, batch_size=args.batch, shuffle=True, num_workers=4)

    #Dataloader Test
    chordSeqDatasetTest = dl.ChordSequencesDatasetClass('../data/preprocessed_data_test.csv', transform=transforms.Compose([dl.ReduChord(args.alphabet), dl.ClassVector(chordsList)]))
    print("Number of test sequences : ", len(chordSeqDatasetTest))
    dataloader_test = DataLoader(chordSeqDatasetTest, batch_size=1, shuffle=True, num_workers=4)

    chordSeqDatasetValid = dl.ChordSequencesDatasetClass('../data/preprocessed_data_validation.csv', transform=transforms.Compose([dl.ReduChord(args.alphabet), dl.ClassVector(chordsList)]))
    print("Number of validation sequences : ", len(chordSeqDatasetValid))
    dataloader_valid = DataLoader(chordSeqDatasetValid, batch_size=1, shuffle=True, num_workers=4)

    print('Dataloader created\n')

    encoder = lstm.EncoderRNN(alpha_size, args.hidden, alpha_size, args.dropout, args.device).to(args.device)
    decoder = lstm.DecoderRNN(args.hidden, alpha_size, alpha_size, args.dropout, args.device).to(args.device)

    loss = []

    for epoch in range(max_epochs):
        print("Epoch ", epoch)

        #Start Training
        print("Start Training")
        encoder.train()
        decoder.train()
        lossep = lstm.trainIters(encoder, decoder, dataloader_train, args.device, epoch/max_epochs,
            print_every=500, plot_every = 2, learning_rate = 0.00001)

        loss = loss + lossep

        #Start Test
        print("Start Test")
        encoder.eval()
        decoder.eval()
        errors,total = evalLSTM.evalIters(encoder, decoder, dataloader_test)
        print(errors/total*100, '% d erreurs')


    torch.save(encoder.state_dict(), args.saveEncoder)
    torch.save(decoder.state_dict(), args.saveDecoder)

    # encoder.load_state_dict(torch.load('gpuresults/encoder_'+args.alphabet+'.dict', map_location = torch.device('cpu')))
    # encoder.eval()
    # decoder.load_state_dict(torch.load('gpuresults/decoder_'+args.alphabet+'.dict', map_location = torch.device('cpu')))
    # decoder.eval()
    # errors,total = evalLSTM.evalIters(encoder, decoder, dataloader_test)
    # print(errors/total*100, '% d erreurs test')
    encoder.eval()
    decoder.eval()
    errors,total = evalLSTM.evalIters(encoder, decoder, dataloader_valid)
    print(errors/total*100, '% d erreurs validation')
