# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 18:17:08 2020

@author: Zitian GAO
"""

import ChordsAlphabets as CA
import ChordsToChromaVectors as CTCV
import numpy as np
import distance

 
'''
deleteDuplicatedElementFromList is for removeing elements in a list which repeat.
'''
def deleteDuplicatedElementFromList(listA):
        #return list(set(listA))
        return sorted(set(listA), key = listA.index)



'''
ACEDeuxChords is for compare the diffence between two chords.
chord1_true : the ground truth chord
chord2_pred : the result predicted
OurAlphabet : the alphabet which is used in trainning 
dis_type : choose which distwnce function you will use in ACE
            == 1 : norm2
            == 2 : tonnetz distance
'''
def ACEDeuxChords(chord1_true, chord2_pred, OurAlphabet, dis_type):

    Notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    OurAlphabet = list(OurAlphabet.values())
    delete_list = ['X']
    OurAlphabet = [x for x in OurAlphabet if x not in delete_list]     
    OurAlphabet = deleteDuplicatedElementFromList(OurAlphabet)
    list_label = []
    for i in Notes:
        for j in OurAlphabet:
            if j != 'N':
                list_label.append(i+str(':')+j)
            
    list_label.append('N')
    #print(list_label)
    
    y_true = np.zeros(len(list_label))
    y_pred = np.zeros(len(list_label))
    
    y_true[list_label.index(chord1_true)] = 1
    y_pred[list_label.index(chord2_pred)] = 1
    
    Vector_list = CTCV.list_mir_label_to_list_vec(list_label,mode='bin_chroma',chroma_mode='bin')


    '''
    Calulate similarity matrix Matrix
    '''
    M =np.zeros((len(Vector_list),len(Vector_list)))
    k = 0.1


    if dis_type == 1: 
        for i in range(len(Vector_list)):
            for j in range(len(Vector_list)):
            #print(Vector_list[i])
            #print(Vector_list[j])
                M[i][j] = 1/(k + np.linalg.norm(np.asarray(Vector_list[i])-np.asarray(Vector_list[j])))
            
        
        M_bar =M/np.linalg.norm(M, ord = np.inf)
    if dis_type == 2:
        for i in range(len(list_label)):
            for j in range(len(list_label)):
                M[i][j] = 1/(k + distance.distance_tonnetz(list_label[i],list_label[j]))
        
        M_bar =M/np.linalg.norm(M, ord = np.inf)
    
    
    '''
    compare y_true_bar with y_pred, here we use norm2
    '''
    y_true_bar = y_true.dot(M_bar)
    result = np.linalg.norm(y_true_bar-y_pred)
    #print(y_true_bar)
    #print(y_pred)
    
    return result


'''

'''

def ACESequence(Chord1Sequence_true, Chord2Sequence_pred, OurAlphabet, dis_type):
    result = 0
    for i in Chord1Sequence_true:
        for j in Chord2Sequence_pred:
            result += ACEDeuxChords(i ,j ,OurAlphabet, dis_type)       
    return result      


#%%
    
# try to use

Score1 = ACEDeuxChords('C:maj', 'E:maj',CA.a1,1)
Score2 = ACESequence(['C:maj', 'C:min'],['E:maj', 'D:min'],CA.a1,1)
Score3 = ACESequence(['C:maj', 'C:min'],['C:maj', 'C:min'],CA.a1,1)
print(Score1,Score2,Score3)

