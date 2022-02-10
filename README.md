# tl_rnn
TL-RNN model: Triplet Loss Recurrent Neural Network

Copyright (C) Stefan Vamosi - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by Stefan Vamosi <stefan@vamosi.org>, September 2021.

Results that were created with this model were recently published by: Stefan Vamosi, Thomas Reutterer and Michael Platzer; A deep recurrent neural network approach to learn sequence similarities for user-identification; Decision Support Systems; https://doi.org/10.1016/j.dss.2021.113718

The TL-RNN code package represents a generic deep neural-network-based framework for quantifying the similarity of ordered sequences in event histories. It combines an LSTM network layer with a triplet loss cost function used for network training. It yields an embedding space that serves as a similarity metric for complex sequential data, can handle multivariate sequential data and incorporate covariates. The motivation 

<figure><img src="images/running_example_seqs.png"><figcaption>The motivation behind the TL-RNN: From image similarity to sequence similarity. Sequential user behavior is like a signature, characteristic patterns distinguish individuals from each other. Taken from (Vamosi, Reutterer, Platzer)</figcaption></figure>


In contrast to approaches that rely on hand-engineered similarity metrics, TL-RNN allows to derive: (i) a purely data-driven sequence similarity metric based on subject-level characteristics, (ii) automatically associating co-occurring events within a sequence (network embedding layer), and (iii) to effectively incorporate any number of covariates for such similarity. Potentially, the model is able to consider all sorts of sequential characteristics, that is specifically: frequency, order, number of events, and co-/occuring event signals.

The triplet loss is based on the idea, that similarity should be learned from co-occurring patterns within a users' history. It is a way to contrast individuals with each other, in order to learn their specific similarities and differences. The triplet learning procedure, in the case of sequences with TL-RNN, is visualized in the following image: 

<figure><img src="images/Sample_Draw_runningexample.png"><figcaption>Triplet comparison: Draw two sequences from the same user and a sequence from a different user. Taken from (Vamosi, Reutterer, Platzer)</figcaption></figure>


The training and prediction is parameterized in the *config.py*. There, the static hyperparameters of the model are set (Not the alpha value that defines the push-pull relation of the triplet loss. This has to be tuned during training). Each user that has at least two existing sequences is used BATCHES_PER_USER x BATCH_SIZE times as an anchor user. In each iteration, a negative sample from another user is drawn randomly to build the triplet. All users are canditates as negatives.

The model consists of an Embedding layer, an LSTM layer trained on a triplet loss function:

<figure><img src="images/Model_Structure.png"><figcaption>Triplet comparison: Draw two sequences from the same user and a sequence from a different user. Taken from (Vamosi, Reutterer, Platzer)</figcaption></figure>


The model weights are shared across the three channels (anchor, positive and negative). After training, a sequence can be translated into an embedding vector, which coordinates reflect its sequential properties. 

The number of dimension of the embedding space object is equal to the number of LSTM cells. The number of cells represents the capacity of the model and always corresponds with the complexity of the data. Although high capacities tend to overfit in general, I suggest to choose a sufficiently large number of weights in order to receive precise positions of sequences. The risk of overfitting is relaxed by the validation loss monitor that is by default used in the training procedure. For the original publication the number of cells was 512, which is very high. Consider that the total number of weights increase non-linearly with the number of cells, due to the fully-connectionism between Embedding and LSTM layer. 

However, the Embedding layer is important to reduce the input vector from a one-hot-encoding with the number of components equal to the cardinality of the input category, to a much lower dimensional real valued vector. Also, the embedding layer is able to capture associations between similar events, e.g. in a customer retail context, the events "Purchase at BP" and "Purchase at Shell" would share strong associations with each other, due to their interchangeability and common appearance within a sequence. Hence, these events would be close inside the Embedding space layer. Please don not confuse this Embedding layer with the embedding ability of the TL-RNN (event embedding vs. sequence embedding).


This can be used to re-identify users, based on behavioral data, or to cluster (segment)
time-series data.

The project is written for categorical input data of the following form:


user_ID     sequence_ID     eventData_1     eventData_2     eventData_3

user_ID -> unique user ID

sequence_ID -> indicates which events belong to the same sequence (a user has several sequences
usually)

eventData_X -> categorical event data, if not already integer encoded, write it into 
COVARIATES_TO_TRANSLATE in config.py

You can see an example run in the run.py, which is considering the attached sequential data

There, a holdout is created and evaluated in an re-identification task on triplets 

Hardware and Software requirements

Make sure you have Python, Keras and Tensorflow running. The project was developed on:

Python 3.7 
Tensorflow 1.5.0

Hardware-wise a GPU was used: CUDA for NVIDIA (TITAN V 12 GB)

If you want to train on CPU, make sure to change "CuDNNLSTM" into "LSTM" inside model.py
