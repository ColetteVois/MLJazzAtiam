#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 19:23:56 2019

@author: stephane
"""

from chordUtil import reduChord
from chordUtil import a0,a1,a2,a3,a5
import numpy as np
from chordUtil import QUALITIES
from collections import Counter
#gladela sam,
#%%
#Equivant à dictbass en chordUtil
Fontamentale_up = {
    'C':1,
    'C#':2,
    'D':3,
    'D#':4,
    'E':5, 
    'F':6,
    'F#':7,
    'G':8,
    'G#':9,
    'A':10,
    'A#':11,
    'B':12}
#%% list of all alphabet

Fontamentale_keys = list(Fontamentale_up)
seperator = ':'
new_listA0 = []
#for i in range(len(Fontamentale_keys)):
#    for j in range(len(listA0)):
#        new_listA0.append(seperator.join([Fontamentale_keys[i],listA0[j]]))
#new_listA0.append('N')



def list_of_alphabet(alphabet):
    theList = list(set(alphabet.values()))
    theList.remove('N')
    new_list = []
    for i in range(len(Fontamentale_keys)):
        for j in range(len(theList)):
            new_list.append(seperator.join([Fontamentale_keys[i],theList[j]]))
    new_list.append('N')
    return new_list


new_listA0 = list_of_alphabet(a0)
new_listA1 = list_of_alphabet(a1)
new_listA2 = list_of_alphabet(a2)
new_listA3 = list_of_alphabet(a3)
new_listA5 = list_of_alphabet(a5)
#%%
Fontamentale_keys = list(Fontamentale_up)
QUALITIES_keys = list(QUALITIES)
seperator = ':'
new_class_keys = []
dict_new_class = {}

for i in range(len(Fontamentale_keys)):
    for j in range(len(QUALITIES_keys)):
        new_class_keys.append(seperator.join([Fontamentale_keys[i],QUALITIES_keys[j]]))
        
for chord in new_class_keys:
    [Fon,qual] = chord.split(":") 
    dict_new_class[chord] = [0] * len(QUALITIES['maj'])
    for k in range(len(QUALITIES[QUALITIES_keys[j]])):
        i = Fontamentale_up[Fon]
        dict_new_class[chord][k] = QUALITIES[qual][( i*12+ k - i + 1) % 12]
    dict_new_class[chord] = str(dict_new_class[chord]) # que pour set
#%% check same values

d = '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]'

chords_nuls =[]
for chord in dict_new_class:
    if dict_new_class[chord] == d:
        print(chord)
        chords_nuls.append(chord)

for i in range(len(chords_nuls)):
    dict_new_class.pop(chords_nuls[i])

dict_new_class["N"] = d 




#%%
cnt = Counter()
dict_values=list(dict_new_class.values())

for word in dict_values:
    cnt[word] += 1
#%%
    
#%%
    
check_chord ='[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]'
for chord in dict_new_class:
    if dict_new_class[chord] == check_chord:
        print(chord)
        
        
#%% list of alphabet
        
listA0 = []
