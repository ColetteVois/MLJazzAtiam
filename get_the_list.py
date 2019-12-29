#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 12:39:38 2019

@author: yujiayang
"""

from chordUtil import a0,a1,a2,a3,a5

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
