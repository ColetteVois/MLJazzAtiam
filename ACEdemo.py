from ChordsAlphabets import *
from ChordsToChromaVectors import *
import numpy as np

def ACEdemo(y_true,y_pred,OurAlphabet):
    
    #y_ture = np.asarray(mir_label_to_bin_chroma_vec(chord1_ture,mode = 'bin'))
    #y_pred = np.asarray(mir_label_to_bin_chroma_vec(chord2_pred,mode = 'bin'))

    Notes = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
    #OurAlphabet = ['maj', 'min']         

    list_label = []
    for i in Notes:
        for j in OurAlphabet:
            if j != 'N':
                list_label.append(i+str(':')+j)
            
    list_label.append('N')
    Vector_list = list_mir_label_to_list_vec(list_label,mode='bin_chroma',chroma_mode='bin')

    M =np.zeros((len(Vector_list),len(Vector_list)))
    k = 0.1


    for i in range(len(Vector_list)):
        for j in range(len(Vector_list)):
            M[i][j] = 1/(k + np.linalg.norm(np.asarray(Vector_list[i])-np.asarray(Vector_list[j])))
        
        M_bar =M/np.linalg.norm(M, ord = np.inf)
    
    y_true_bar = y_true*M_bar
    result = np.linalg.norm(y_true_bar-y_pred)
    
    return result



#%%
    
# try to use
y_true = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
y_pred = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
Score = ACEdemo(y_true, y_pred,['maj', 'min', 'N'])
print(Score)