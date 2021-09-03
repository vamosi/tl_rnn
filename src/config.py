"""
TL-RNN Triplet Loss Recurrent Neural Networks - a sequence embedder for discrete time-series data-sets
for segmentation or re-identification of human behavior

Author: Stefan Vamosi
"""

from keras.optimizers import Adam

"""
main config file for tl-rnn model where main parameters are set and handed to dataprep and model classes
"""


"""
specifiy training parameters like optimizer and learning rate
epochs is the maximal number of epochs proceeded
sequence_len_max is the maximal sequence legnth modeled, if sequence exeeds this lengths 
-> the last actions are taken
"""
OPTIMIZER = Adam(lr=0.0001)
EPOCHS = 12
SEQUENCE_LEN_MAX = 100 #maximal length in the template dataset is 89, so 100 is fine


"""
patience defines the number of epochs that have to 
improve val-loss before early stopping is taking place 
"""
PATIENCE = 4


"""
batches_per_user defines how many batches of size batch_size should be sampled per user
batch_size defines how many samples (triplets) are considered for one back-propagation calculation
This means, at least batch_size samples are considered per user (batches_per_user = 1)
"""
BATCHES_PER_USER = 2
BATCH_SIZE = 64


"""
if a sequence is longer than SEQUENCE_LEN_MAX, this setting defines what to do:
if TAKE_LAST_ACTIONS = True -> take the SEQUENCE_LEN_MAX last observations and throw away the first ones
if TAKE_LAST_ACTIONS = False [default] -> take the first observations and throw away the last ones
"""
TAKE_LAST_ACTIONS = False


"""
max_cardinality is a cap for the maximal cardinality that for each feature is used
should be limited in certain projects to limit the model size
consider that super rare signals do not contribute to a comparative loss anyways, as they are so rare
"""
MAX_CARDINALITY = 100000


"""
if using validation error supervision, this valid_split variable defines
how much of the original data are not used for training but for validation
"""
VALID_SPLIT = 0.05


""" basic models path data """
MODEL_PATH = "./logs/"
TENSORBOARD_PATH = "tensorboard/"


"""
define in the list all column names you want to use for this project
consider that all these columns are used for triplet comparison learning
"""
COVARIATES_ALL = ["eventData_1", "eventData_2", "eventData_3"]


"""
define covariates_to_translate into numerical (integer) categories, 
for example text strings, etc.
do this by rank-based encoding, as this is transparent to see the rarity of signals immediately
ATTENTION: ALL VARIABLES ARE ENCODED CATEGORICAL
"""
COVARIATES_TO_TRANSLATE = ["eventData_1"]


"""
calculate the number of total covariates and the number of categorical covariates
"""
COV_NUMBER = list(range(len(COVARIATES_ALL)))