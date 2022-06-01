# Short script used to create the blosum matrices necessary when using "use_blosum" argument
import pickle
import numpy as np
from sklearn.decomposition import PCA as PCA
alphabet='ARNDCEQGHILKMFPSTWYVX'

def subsmatFromAA2(identifier, data_file='aaindex2.txt'):
     """Retrieve a substitution matrix from AAindex2-file, scale it between 0 and 1, and add gap"""
     with open(data_file, 'r') as f:
          for line in f:
               if identifier in line:
                    break
          for line in f:
               if line[0] == 'M':
                    split_line = line.replace(',', ' ').split()
                    rows = split_line[3]
                    cols = split_line[6]
                    break

          subsmat = np.zeros((21, 21), dtype=np.float64)
          i0 = 0
          for line in f:
               i = alphabet.find(rows[i0])
               vals = line.split()
               for j0 in range(len(vals)):
                    j = alphabet.find(cols[j0])
                    subsmat[i, j] = vals[j0]
                    subsmat[j, i] = vals[j0]
               i0 += 1
               if i0 >= len(rows):
                    break
     subsmat[:-1, :-1] += np.abs(np.min(subsmat)) + 1
     subsmat[:-1, :-1] /= np.max(subsmat)
     subsmat[-1, -1] = np.min(np.diag(subsmat)[:-1])

     return subsmat

def get_pcs(subsmat,d):
    """Get first d pca-components from the given substitution matrix."""
    pca = PCA(d)
    pca.fit(subsmat)
    pc = pca.components_
    return pc

def encode_with_pc(seq_lists, lmaxes, pc):
    """ Encode the sequence lists (given as numbers), with the given pc components (or other features)
    lmaxes contains the maximum lengths of the given sequences. """
    d = pc.shape[0]
    X = np.zeros((len(seq_lists[0]),d*sum(lmaxes)))
    i_start, i_end = 0, 0
    for i in range(len(seq_lists)):
        Di = d*lmaxes[i]
        i_end += Di
        for j in range(len(seq_lists[i])):
            X[j,i_start:i_end] = np.transpose( np.reshape( np.transpose( pc[:,seq_lists[i][j]] ), (Di,1) ) )
        i_start=i_end
    return X
a = subsmatFromAA2("MUET020102")
pca_blosum = get_pcs(a, 16)
pca_blosum = pca_blosum.transpose(1,0)
blusum_dict = {}
for ind_,l  in enumerate(alphabet):
     blusum_dict[l] = pca_blosum[ind_]
# PERSONALIZE (PCA); maybe use one-hot
pickle.dump(blusum_dict,open("blosum.bin", "wb"))
