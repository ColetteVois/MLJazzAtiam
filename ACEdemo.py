import ChordsAlphabets as CA
import ChordsToChromaVectors as CTCV
import numpy as np

def deleteDuplicatedElementFromList(listA):
        #return list(set(listA))
        return sorted(set(listA), key = listA.index)


def ACEdemo(chord1_ture,chord2_pred,OurAlphabet):

    Notes = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
    
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
    
    y_true[list_label.index(chord1_ture)] = 1
    y_pred[list_label.index(chord2_pred)] = 1
    
    Vector_list = CTCV.list_mir_label_to_list_vec(list_label,mode='bin_chroma',chroma_mode='bin')

    M =np.zeros((len(Vector_list),len(Vector_list)))
    k = 0.1


    for i in range(len(Vector_list)):
        for j in range(len(Vector_list)):
            #print(Vector_list[i])
            #print(Vector_list[j])
            M[i][j] = 1/(k + np.linalg.norm(np.asarray(Vector_list[i])-np.asarray(Vector_list[j])))
        
        M_bar =M/np.linalg.norm(M, ord = np.inf)
    
    #print(y_true)
    y_true_bar = y_true.dot(M_bar)
    result = np.linalg.norm(y_true_bar-y_pred)
    #print(y_true_bar)
    #print(y_pred)
    
    return result



#%%
    
# try to use

Score1 = ACEdemo('C:maj', 'E:maj',CA.a5)
Score2 = ACEdemo('C:maj', 'C:min',CA.a5)
Score3 = ACEdemo('C:maj', 'D:dim7',CA.a5)
Score4 = ACEdemo('C:maj', 'C:maj',CA.a5)
print(Score1,Score2, Score3, Score4)

