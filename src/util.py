from keras.optimizers import Adam
import _pickle as pickle
import numpy as np


def datatofile(data,filename):
    f=open(filename, 'wb')
    pickle.dump(data, f)
    f.close()
    return f

def datafromfile(filename):
    f=open(filename, 'rb')
    return pickle.load(f)

def embedding_dim_heuristic(category_count: int) -> int:
   """
   Returns a heuristic value for the embedding dimension.
   :param category_count: number of categories for column
   :returns: estimated embedding dimension
   """

   return int(np.ceil(category_count**(1/4))*3)



