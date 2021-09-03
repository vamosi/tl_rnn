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


"""
define directory for data
"""
directory = "~/tl_rnn/data/"

################################### START TRAINING DATA PROCESSING ##########################################

"""
read column-like dataframe and get rid of NA's
"""
df = pd.read_csv(directory + "sequential_data_train.csv", verbose=True)
df = df.fillna(0)

"""
create dataprep object where all
"""
dataobj = dataprep(dataframe=df, id_name="user_ID", sequence_id = "sequence_ID")

"""
run preprocessor and bring datframe into propper list-like data structure for sampler
"""
dataobj.preprocessor()

"""
evaluate cardinalities, consider that all inputs will be modeled as categorical variables
"""
dataobj.build_cardinality()


################################### START TEST DATA PROCESSING ##########################################

"""
read column-like dataframe and get rid of NA's
"""
df_test = pd.read_csv(directory + "sequential_data_test.csv")
df_test = df_test.fillna(0)

"""
create dataprep object where all
"""
dataobj_test = dataprep(dataframe=df_test, id_name="user_ID", sequence_id = "sequence_ID")

"""
run preprocessor and bring datframe into propper list-like data structure for sampler
"""
dataobj_test.preprocessor()

"""
sample triplets from holdout (test) set to test re-identification score
"""
dataobj_test.sampler(triplets_per_user=8, users_per_file="all")



################################### START TRAINING PROCESS ##########################################


"""
define cells and build tl-rnn keras model
"""
my_tlrnn = tlrnn(cells=128, dataobj=dataobj)
my_tlrnn.build()


"""
train model
tune alpha: seperation strength (regularizer) for triplet seperation
too much: model doesn't generalize well
too weak: model doesn't discriminate well
"""
alpha=1.0

run_name = "my_run_01"


"""
generator model: draws samples from dataobj automatically
define test_set manually from test set above for re-identification evaluation on hold-out
"""
my_tlrnn.train_generator(run_name=run_name, alpha=alpha, test_set=dataobj_test.triplet_tensor)