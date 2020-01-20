# ATTENTION： When you use this function, please make sure every file in the input Folder
#             is a .xlab file. (Remove 'READMD.txt' in the 'jazz_xlab' folder)

import numpy as np
import os


def preProcessing(FolderName):
    jazz_xlab = os.listdir(FolderName)# FolderName is a string which is called 'jazz_xlab'
    csvFile = open(FolderName+'../preprocessed_data.csv', 'w')

    n = 1
    for MusicPiece in jazz_xlab :
        ExtraBeatChord = []
        preSequence = []
        ChordSequence = []
        # read every .xlab file and split the content in line
        file = open(FolderName+MusicPiece, 'r')
        Content = file.read()
        file.close()
        Content = Content.split('\n')

        # only save the beat and Chord imformations
        for i in range(len(Content)-1):
            CurrentLine = Content[i+1].split(' ')
            CurrentBeat = int((CurrentLine[2]))
            CurrentChord = CurrentLine[4]
            ExtraBeatChord.append([CurrentBeat,CurrentChord])

        Tonality = CurrentLine[::-1][0];

        # write the Chords in a list respect their beat number
        for j in range(len(ExtraBeatChord)):
            for k in range(ExtraBeatChord[j][0]):
                preSequence.append(ExtraBeatChord[j][1])

        #map(+str(Tonality),preSequence)

        # re-write the chord sequence as 16 chords in every line
        for i in range(len(preSequence)):
            if i+16 <= len(preSequence):
                temp = preSequence[i:i+16]
                for j in range(16) :
                    if (j%4 + 1 +i%4)%4 == 0:
                        temp[j] = temp[j] +str('%') + str(4)+str('%')+str(Tonality)
                    else:
                        temp[j] = temp[j] +str('%') + str(((j%4 + 1 +i%4))%4)+str('%')+str(Tonality)

                for k in range(15):
                    csvFile.write(temp[k]+';')
                csvFile.write(temp[15] + '\n')
    csvFile.close()

ChordSequence = preProcessing('../data/train/')
os.rename(r'../data/preprocessed_data.csv',r'../data/preprocessed_data_train.csv')
ChordSequence = preProcessing('../data/test/')
os.rename(r'../data/preprocessed_data.csv',r'../data/preprocessed_data_test.csv')
ChordSequence = preProcessing('../data/validation/')
os.rename(r'../data/preprocessed_data.csv',r'../data/preprocessed_data_validation.csv')
#print(ChordSequence)
