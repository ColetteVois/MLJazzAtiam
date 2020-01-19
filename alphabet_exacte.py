#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 12:52:11 2019

@author: yujia
"""
#Class (sans reduction)

import difflib
import numpy as np
All_chords = {'[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]': 'C:maj',
         '[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]': 'C:min',
         '[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]': 'C:aug', #E:aug G#:aug
         '[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]': 'C:dim',
         '[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]': 'C:sus4',#F:sus2
         '[1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]': 'C:sus2', #G:sus4
         '[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]': 'C:7', #C:9 C:b9 C:#9 C:11 C:#11 C:13 C:b13
         '[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]': 'C:maj7',# C:maj9 C:maj13
         '[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]': 'C:min7',#C:min9 C:min11 C:min13 D#:maj6
         '[1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]': 'C:minmaj7',
         '[1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]': 'C:maj6', #A:min7 A:min9 A:min11 A:min13
         '[1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0]': 'C:min6', #A:hdim7
         '[1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]': 'C:dim7',#D#:dim7 F#:dim7 A:dim7
         '[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]': 'C:hdim7',#D#:min6
         '[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': 'C:1',
         '[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]': 'C:5',
         '[0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]': 'C#:maj',
         '[0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]': 'C#:min',
         '[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]': 'C#:aug',#F:aug A:aug
         '[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]': 'C#:dim',
         '[0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]': 'C#:sus4',#F#:sus2
         '[0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]': 'C#:sus2',#G#:sus4
         '[0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]': 'C#:7',#C#:9 C#:b9 C#:#9 C#:11 C#:#11 C#:13 C#:b13
         '[1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]': 'C#:maj7',#C#:maj9 C#:maj13
         '[0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]': 'C#:min7', #C#:min9 C#:min11 C#:min13 E:maj6
         '[1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]': 'C#:minmaj7',
         '[0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0]': 'C#:maj6',#A#:min7 A#:min9 A#:min11 A#:min13
         '[0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0]': 'C#:min6',#A#:hdim7
         '[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]': 'C#:dim7', #E:dim7 G:dim7 A#:dim7
         '[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]': 'C#:hdim7', #E:min6
         '[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': 'C#:1',
         '[0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]': 'C#:5',
         '[0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]': 'D:maj',
         '[0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]': 'D:min',
         '[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]': 'D:aug',#F#:aug A#:aug
         '[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]': 'D:dim',
         '[0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0]': 'D:sus4', #G:sus2
         '[0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0]': 'D:sus2', #A:sus4
         '[1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]': 'D:7',#D:9 D:b9 D:#9 D:11 D:#11 D:13 D:b13
         '[0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]': 'D:maj7',#D:maj9 D:maj13
         '[1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]': 'D:min7', #D:min9 D:min11 D:min13 F:maj6
         '[0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0]': 'D:minmaj7',
         '[0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1]': 'D:maj6', #B:min7 B:min9 B:min11 B:min13
         '[0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1]': 'D:min6', #B:hdim7
         '[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]': 'D:dim7', #F:dim7 G#:dim7 B:dim7
         '[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]': 'D:hdim7', #F:min6
         '[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]': 'D:1',
         '[0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]': 'D:5',
         '[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]': 'D#:maj',
         '[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]': 'D#:min',
         '[0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]': 'D#:aug', #G:aug B:aug
         '[0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]': 'D#:dim',
         '[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0]': 'D#:sus4', #G#:sus2
         '[0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0]': 'D#:sus2', #A#:sus4
         '[0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]': 'D#:7', #D#:9 D#:b9 D#:#9 D#:11 D#:#11 D#:13 D#:b13
         '[0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0]': 'D#:maj7', #D#:maj9 D#:maj13
         '[0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]': 'D#:min7', #D#:min9 D#:min11 D#:min13 F#:maj6
         '[0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0]': 'D#:minmaj7',
         '[0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]': 'D#:hdim7', #F#:min6
         '[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]': 'D#:1',
         '[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]': 'D#:5',
         '[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]': 'E:maj',
         '[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]': 'E:min',
         '[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]': 'E:dim',
         '[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1]': 'E:sus4', #A:sus2
         '[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1]': 'E:sus2', #B:sus4
         '[0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1]': 'E:7', #E:9 E:b9 E:#9 E:11 E:#11 E:13 E:b13
         '[0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1]': 'E:maj7',#E:maj9 E:maj13
         '[0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1]': 'E:min7', #E:min9 E:min11 E:min13 G:maj6
         '[0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1]': 'E:minmaj7',
         '[0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]': 'E:hdim7', #G:min6
         '[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]': 'E:1',
         '[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]': 'E:5',
         '[1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]': 'F:maj',
         '[1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]': 'F:min',
         '[0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1]': 'F:dim',
         '[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]': 'F:sus4', #A#:sus2
         '[1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0]': 'F:7', #F:9 F:b9 F:#9 F:11 F:#11 F:13 F:b13
         '[1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0]': 'F:maj7', #F:maj9 F:maj13
         '[1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]': 'F:min7', #F:min9 F:min11 F:min13 G#:maj6
         '[1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0]': 'F:minmaj7',
         '[0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]': 'F:hdim7', #G#:min6
         '[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]': 'F:1',
         '[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]': 'F:5',
         '[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]': 'F#:maj',
         '[0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]': 'F#:min',
         '[1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]': 'F#:dim',
         '[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]': 'F#:sus4', #B:sus2
         '[0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0]': 'F#:7', #F#:9 F#:b9 F#:#9 F#:11 F#:#11 F#:13 F#:b13
         '[0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]': 'F#:maj7', #F#:maj9 F#:maj13
         '[0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]': 'F#:min7', #F#:min9 F#:min11 F#:min13 A:maj6
         '[0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]': 'F#:minmaj7',
         '[1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]': 'F#:hdim7', #A:min6
         '[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]': 'F#:1',
         '[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]': 'F#:5',
         '[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]': 'G:maj',
         '[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]': 'G:min',
         '[0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]': 'G:dim',
         '[0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1]': 'G:7', #G:9 G:b9 G:#9 G:11 G:#11 G:13 G:b13
         '[0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1]': 'G:maj7',#G:maj9 G:maj13
         '[0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]': 'G:min7', #G:min9 G:min11 G:min13 A#:maj6
         '[0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0]': 'G:minmaj7',
         '[0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0]': 'G:hdim7', #A#:min6
         '[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]': 'G:1',
         '[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]': 'G:5',
         '[1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]': 'G#:maj',
         '[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]': 'G#:min',
         '[0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1]': 'G#:dim',
         '[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0]': 'G#:7', #G#:9 G#:b9 G#:#9 G#:11 G#:#11 G#:13 G#:b13
         '[1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0]': 'G#:maj7', #G#:maj9 G#:maj13
         '[0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]': 'G#:min7', #G#:min9 G#:min11 G#:min13 B:maj6
         '[0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1]': 'G#:minmaj7',
         '[0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1]': 'G#:hdim7', #B:min6
         '[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]': 'G#:1',
         '[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]': 'G#:5',
         '[0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]': 'A:maj',
         '[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]': 'A:min',
         '[1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]': 'A:dim',
         '[0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]': 'A:7', #A:9 A:b9 A:#9 A:11 A:#11 A:13 A:b13
         '[0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]': 'A:maj7', #A:maj9 A:maj13
         '[1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0]': 'A:minmaj7',
         '[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]': 'A:1',
         '[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]': 'A:5',
         '[0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]': 'A#:maj',
         '[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]': 'A#:min',
         '[0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]': 'A#:dim',
         '[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0]': 'A#:7', #A#:9 A#:b9 A#:#9 A#:11 A#:#11 A#:13 A#:b13
         '[0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0]': 'A#:maj7', #A#:maj9 A#:maj13
         '[0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0]': 'A#:minmaj7',
         '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]': 'A#:1',
         '[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]': 'A#:5',
         '[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1]': 'B:maj',
         '[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]': 'B:min',
         '[0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1]': 'B:dim',
         '[0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]': 'B:7', #B:9 B:b9 B:#9 B:11 B:#11 B:13 B:b13
         '[0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1]': 'B:maj7', #B:maj9 B:maj13
         '[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1]': 'B:minmaj7',
         '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]': 'B:1',
         '[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]': 'B:5',
         '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]': 'N'
         }

list_exacte = list(All_chords.values())
         

