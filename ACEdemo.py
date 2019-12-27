from ChordsAlphabets import *
from ChordsToChromaVectors import *
import numpy as np

def ACEdemo(chord1_ture,chord2_pred):
    
    y_ture = np.asarray(mir_label_to_bin_chroma_vec(chord1_ture,mode = 'bin'))
    y_pred = np.asarray(mir_label_to_bin_chroma_vec(chord2_pred,mode = 'bin'))

    Notes = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
    OurAlphabet = ['maj', 'min']         

    list_label = []
    for i in Notes:
        for j in OurAlphabet:
            list_label.append(i+str(':')+j)

    Vector_list = list_mir_label_to_list_vec(list_label,mode='bin_chroma',chroma_mode='bin')

    M =np.zeros((len(Vector_list),len(Vector_list)))
    k = 0.001


    for i in range(len(Vector_list)):
        for j in range(len(Vector_list)):
            M[i][j] = 1/(k + np.linalg.norm(np.asarray(Vector_list[i])-np.asarray(Vector_list[j])))
        
        M_bar =M/np.linalg.norm(M, ord = np.inf)
    
    y_ture_bar = y_ture*M_bar
    result = np.linalg.norm(y_ture_bar-y_pred)
    
    return result

#%%
    
# try to use
    
Score = ACEdemo('C:maj','D:min')
