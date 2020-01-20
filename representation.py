# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:08:27 2020

@author: Zitian
"""
import chordUtil as CU
import numpy as np
import DictionaryForRepresentation
def representation(chord, rho):
    V = np.zeros(12)
    clavier = {
                'C'  : 0,
                'C#' : 1,
                'D'  : 2,
                'D#' : 3,
                'E'  : 4,
                'F'  : 5,
                'F#' : 6,
                'G'  : 7,
                'G#' : 8,
                'A'  : 9,
                'A#' : 10,
                'B'  : 11
                }
    if chord == '':
        return V
    else:
        temp = chord.split(':')
        note = temp[0]
        ChordType = temp[1]
        V[clavier[note]] = rho
        newroot = note
        number =  clavier[newroot]
        for i in DictionaryForRepresentation.interval[ChordType]:
            if i > 0:
                if (i + number) < 12:
                    #print('Here')
                    for j in range(i):
                        newroot = CU.tr[newroot]
                    V[clavier[newroot]] += rho
                elif (i + number) < 24:
                    
                    #print(number)
                    for j in range(i):
                        newroot = CU.tr[newroot]
                    V[clavier[newroot]] += (rho*rho)
                elif (i + number) >= 24:
                    number = i + number
                    #print(number)
                    for j in range(i):
                        newroot = CU.tr[newroot]
                    V[clavier[newroot]] += (rho*rho*rho*rho)
                number = i + number
        return V

def DistanceByRepresentation(Chord1,Chord2,rho):
      temp = representation(Chord1, rho)-representation(Chord2, rho)  
      return  np.linalg.norm(temp)
#%%
#
'''
A = 'B:maj13'
B = 'E:min'
result = DistanceByRepresentation(A,B, 0.5)
print(result)
'''