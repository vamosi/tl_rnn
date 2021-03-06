"""
TL-RNN Triplet Loss Recurrent Neural Networks - a sequence embedder for discrete time-series data-sets
for segmentation or re-identification of human behavior

Author: Stefan Vamosi
"""

from util import *
import pickle
import pandas as pd
import random
from model import tlrnn
from dataPrep import dataprep
import pickle
import tensorflow as tf


print("tensorflow:")
tf.test.is_gpu_available( cuda_only=False, min_cuda_compute_capability=None )
print("end tensorflow")

"""
define directory for data
"""
directory = "~/tl_rnn/data/"

################################### START TRAINING DATA PROCESSING ##########################################

"""
read dataframe and get rid of NA's
"""
df = pd.read_csv(directory + "sequential_data_train.csv", verbose=True)
df = df.fillna(0)

df = df.iloc[:50000,:] ############ !!!!!!! test
"""
create dataprep object. in this object all hyperparameters for dataprocessing and sampling are stored
please double check all parameters in config.py before running it
"""
dataobj = dataprep(dataframe=df, id_name="user_ID", sequence_id = "sequence_ID")


"""
run preprocessor and bring datframe into propper list-like data structure for sampler
each list element is a sequence
"""
dataobj.preprocessor()


"""
evaluate cardinalities of input values
"""
dataobj.build_cardinality()


################################### START TEST DATA PROCESSING ##########################################

"""
read column-like dataframe and get rid of NA's
"""
df_test = pd.read_csv(directory + "sequential_data_test.csv")
df_test = df_test.fillna(0)

df_test = df_test[:50000] #!!!!!!!!!!!!!!!!! test


"""
create dataprep object for model test validation
"""
dataobj_test = dataprep(dataframe=df_test, id_name="user_ID", sequence_id = "sequence_ID")


"""
run preprocessor and bring datframe into propper list-like data structure for sampler
"""
dataobj_test.preprocessor()


"""
VERY IMPORTANT!!

Overwrite the feature encodings (rank-based) with the ones from the training set
Otherwise training and test are encoded differently and the predicitve power gets bad
"""
dataobj_test.ranks = dataobj.ranks


"""
sample triplets from holdout (test) set to test re-identification score
"""
dataobj_test.sampler(triplets_per_user=8, users_per_file="all")



################################### START TRAINING PROCESS ##########################################


"""
define cells and build tl-rnn keras model
put in dataobj, where also the pre-processed data-set
is included for "online" generator sampling
"""
my_tlrnn = tlrnn(cells=128, dataobj=dataobj)
my_tlrnn.build()


"""
alpha: seperation strength (regularizer) for triplet seperation
too much: model doesn't generalize well
too weak: model doesn't discriminate well
start with alpha=1.0 and work upwards
"""
alpha=1.0
run_name = "my_run_01"



"""
Now, let's embedd a sequence into an embedding vector with length of cells:
"""
# take a single sequence out of the holdout data set
single_seq = dataobj_test.data_list[0][0]
# check if sequence is not longer than our model was trained for
if len(single_seq[:,0]) > my_tlrnn.sequence_len_max:
    raise Exception("Input sequence is longer than seq-len-max. Shorten the sequence and repeat.")

"""
zero-padding of the sequence to bring it to the nominal sequence length
first dim is important, but for the single sequence case just 1
"""
seq = np.zeros((1, dataobj.sequence_len_max, my_tlrnn.cov_num), dtype='int32')
seq[0, :len(single_seq[:,0]), :] = single_seq[:, :]

"""
translate the sequence into embedding vector with the trained tl_rnn model
model requires list of features
single_embedd is your embedding vector
"""
single_embedd = my_tlrnn.model_pred.predict([seq[:, :, (i):(i+1)] for i in list(range(my_tlrnn.cov_num))])

