"""
TL-RNN Triplet Loss Recurrent Neural Networks - a sequence embedder for discrete time-series data-sets
for segmentation or re-identification of human behavior

Author: Stefan Vamosi
"""

import keras
from keras.engine import Model
from keras.layers import *
from keras.callbacks import TensorBoard, LearningRateScheduler, EarlyStopping, ModelCheckpoint

#!!!!!!!!!!important!!!!!!!!!!!!!!!!
# here set config file (config, config_Gfk, etc.)
from config import OPTIMIZER, EPOCHS, SEQUENCE_LEN_MAX, \
                   MODEL_PATH, BATCH_SIZE, \
                   TENSORBOARD_PATH, PATIENCE, VALID_SPLIT, BATCHES_PER_USER, TAKE_LAST_ACTIONS
import keras.backend as k
import tensorflow as tf
import numpy as np
import os
from util import embedding_dim_heuristic
import random


class MyCallback_test(keras.callbacks.Callback):
    """
    Defining callback to be executed after each batch
    """
    def __init__(self, tlrnn_object, test_set):
        self.test_set = test_set
        self.tlrnn_object = tlrnn_object
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.tlrnn_object.evaluate(triplet_tensor=self.test_set)

def compute_euclidean_distance(x, y, axis_to_reduce):
    """
    Computes the euclidean distance between two tensorflow variables
    """
    d = tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=axis_to_reduce)
    return d


def compute_manhatten_distance(x, y, axis_to_reduce):
    """
    Computes the euclidean distance between two tensorflow variables
    """
    d = tf.reduce_sum(tf.abs(tf.subtract(x, y)), axis=axis_to_reduce)
    return d


def compute_cosine_similarity(x, y, axis_to_reduce):
    """
    Computes the cosine similarity norm -> d = A*B/(Norm(A)*Norm(B)
    """
    nominator = tf.reduce_sum(tf.multiply(x, y), axis=axis_to_reduce)
    denominator = tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis_to_reduce))*tf.sqrt(tf.reduce_sum(tf.square(y), axis=axis_to_reduce))
    d = nominator/denominator
    return d


def loss_wrapper(alpha, beta, gamma, cells):
    """
    I proudly present, the triplet loss which has to be wrapped around a loss wrapper
    """
    def triplet_loss(y_true, y_pred):
        distance_0_plus = compute_manhatten_distance(y_pred[:, 0:cells],
                                                     y_pred[:, cells:(2*cells)], axis_to_reduce=1)
        distance_0_minus = compute_manhatten_distance(y_pred[:, 0:cells],
                                                      y_pred[:, (2*cells):(3*cells)], axis_to_reduce=1)
        result = tf.maximum(((gamma*distance_0_plus) - (beta*distance_0_minus) + alpha), 0)
        #mask = k.less(result, alpha)
        #result = tf.multiply(result, tf.cast(mask, tf.float32))
        #inverted_mask = tf.logical_not(mask)
        #to_add = tf.multiply(tf.cast(alpha, tf.float32), tf.cast(inverted_mask, tf.float32))
        #result = result + to_add
        return result
    return triplet_loss


class tlrnn():
    """
    main tl-rnn class to build the model. Important members are:
    self.covariates: covariates used for model, input dataprep.covariates, includes also cardinalities!
    self.sequence_len_max: maximal sequence length the model considers, no matter how long are the actual sequences
    self.take_last_actions: crop either from back of the sequence [True] or from beginning [False]
    """

    def __init__(self, cells, dataobj):
        self.data_list = dataobj.data_list
        self.covariates = dataobj.cardinalities
        self.cov_names = dataobj.cardinalities.keys()
        self.cov_dims = dataobj.cardinalities.values()
        self.cov_num = len(dataobj.cardinalities)
        self.sequence_len_max = SEQUENCE_LEN_MAX
        self.take_last_actions = TAKE_LAST_ACTIONS
        self.cells = cells
        self.model_train = Model()
        self.model_pred = Model()
        self.epoch_max = EPOCHS
        self.model_path = MODEL_PATH
        self.tensorboard_path = TENSORBOARD_PATH
        self.patience = PATIENCE
        self.batch_size = BATCH_SIZE
        self.valid_split = VALID_SPLIT
        self.batches_per_user = BATCHES_PER_USER
        self.beta = 1.0 # symmetrical push-pull relation
        self.gamma = 1.0 # symmetrical push-pull relation
        self.optimizer = OPTIMIZER
        self.score_L1 = None
        self.score_L2 = None
        print("COVARIATES: ", self.covariates)
        print("SEQUENCE LEN MAX", self.sequence_len_max)
        print("CELLS", self.cells)
        print("PATIENCE", self.patience)
        print("BATCH SIZE", self.batch_size)
        print("BATCHES PER USER", self.batches_per_user)
        print("OPTIMIZER", self.optimizer)

    def train_valid_split(self, data_list):
        """
        in generator mode prepare data-sets for training and validation to feed it into triplet_generator
        remember, this validation is just for the sake of early stopping (Epoch control)
        ......
        """
        random.shuffle(data_list)
        cut = int(len(data_list) * self.valid_split)
        train_set = data_list[cut:]
        valid_set = data_list[:cut]

        return train_set, valid_set

    def triplet_generator_list_of_lists_cutoff(self, data_list):
        """
        generator for generator training, this generator yields at least one batch_size per anchor user
        if the user has at least two sequences, otherwise the user is skipped and only a candidate for
        a negative sequence. if the sequence is longer that sequence_len_max, the last valid actions are
        considered.
        return values:
        it prints the number of users that were skipped, because they had only one sequence
        yields a list for one trainings batch
        """
        # prepare triplet tensor to fed into model training
        user_list = list(range(0, len(data_list)))
        while True:
            skipping_count = 0
            # iterate through all users
            for indx_user, user in enumerate(data_list):
                # check if user has to sequences to build anchor-positive pair
                if len(user) > 1:
                    # yield for each user "batches_per_user" often times a batch
                    for _ in range(self.batches_per_user):
                        triplet_tensor = np.zeros((self.batch_size, self.sequence_len_max, 3 * len(self.covariates)), dtype='int32')
                        # sample triplets_per_user triplets
                        for sample_indx in range(self.batch_size):
                            # draw anchor- and positive-sequence
                            anchor, positive = random.sample(list(user), 2)
                            # exclude anchor user and draw negative user
                            user_list_pop = user_list.copy()
                            user_list_pop.pop(indx_user)
                            neg_indx = random.choice(user_list_pop)
                            # draw negative-sequence
                            negative = random.sample(list(data_list[neg_indx]), 1)[0]
                            # append lists
                            if self.take_last_actions:
                                if len(anchor[:, 0]) > self.sequence_len_max:
                                    anchor = anchor[(len(anchor[:, 0])-self.sequence_len_max):, :]
                                if len(positive[:, 0]) > self.sequence_len_max:
                                    positive = positive[(len(positive[:, 0])-self.sequence_len_max):, :]
                                if len(negative[:, 0]) > self.sequence_len_max:
                                    negative = negative[(len(negative[:, 0])-self.sequence_len_max):, :]
                            else:
                                if len(anchor[:, 0]) > self.sequence_len_max:
                                    anchor = anchor[:self.sequence_len_max, :]

                                if len(positive[:, 0]) > self.sequence_len_max:
                                    positive = positive[:self.sequence_len_max, :]

                                if len(negative[:, 0]) > self.sequence_len_max:
                                    negative = negative[:self.sequence_len_max, :]
                            triplet_tensor[sample_indx, (self.sequence_len_max-len(anchor[:,0])):, 0:len(self.covariates)] = anchor
                            triplet_tensor[sample_indx, (self.sequence_len_max-len(positive[:,0])):, len(self.covariates):(2 * len(self.covariates))] = positive
                            triplet_tensor[sample_indx, (self.sequence_len_max-len(negative[:,0])):, (2 * len(self.covariates)):(3 * len(self.covariates))] = negative

                        input_list = [triplet_tensor[:, :, idx:idx + 1] for idx in range(3 * self.cov_num)]
                        dummy_target = np.zeros(self.batch_size)
                        yield (input_list, dummy_target)
                else:
                    skipping_count += 1
                    pass

            print(" number of users skipped: ", skipping_count)

    def build(self):
        """
        Method to build the model architecture.
        The layer dimensions are evaluated and created automatically accoring the config file
        The current set-up is considering training on GPU (CuDNNLSTM), change it accordingly to your
        set-up. The complete layer structure for anchor, positive and negative samples are re-used for each of them
        -> shared layer
        """
        # replace self.sequence_length with None
        output_dims = [embedding_dim_heuristic(dim) for dim in list(self.cov_dims)]
        input_layer = [Input(shape=(self.sequence_len_max, 1)) for _ in range(3*self.cov_num)] #[Input(shape=(None, 1)) for _ in range(3*self.cov_num)]
        embedding_layer = [TimeDistributed(Embedding(output_dim=output_dims[idx], input_dim=list(self.cov_dims)[idx])) for idx in range(self.cov_num)]

        connected_layer = []

        for idx, val in enumerate(input_layer):
            connected_layer.append(embedding_layer[idx%self.cov_num](val))

        reshaped_layer = [Reshape((-1, output_dims[idx%self.cov_num]),
                                  name = "reshape_0"+str(idx))(val) for idx, val in enumerate(connected_layer)]

        if self.cov_num==1:
            anchor = reshaped_layer[0:self.cov_num]
            pos = reshaped_layer[self.cov_num:(2*self.cov_num)]
            neg = reshaped_layer[(2*self.cov_num):(3*self.cov_num)]
        else:
            anchor = concatenate(reshaped_layer[0:self.cov_num])
            pos = concatenate(reshaped_layer[self.cov_num:(2*self.cov_num)])
            neg = concatenate(reshaped_layer[(2*self.cov_num):(3*self.cov_num)])

        lstm1 = CuDNNLSTM(units=self.cells, stateful=False, return_sequences=True, name="lstm_gpu", return_state=True)

        out1, state_h1, state_c1 = lstm1(anchor)
        out2, state_h2, state_c2 = lstm1(pos)
        out3, state_h3, state_c3 = lstm1(neg)

        out = concatenate([state_c1, state_c2, state_c3])

        self.model_train = Model(input_layer, [out])
        self.model_pred = Model(input_layer[0:self.cov_num], [state_c1])

    def load(self, path):
        """
        load pre-trained model
        """
        self.model_pred.load_weights(path)

    def train(self, run_name=None, tensor_train=None, alpha=None):
        """
        Standard training scheme: read in one training tensor tensor_train created by dataprep.sampler
        In the current setup all epochs are saved to a model file without validation loss. Can be tested manually
        """
        print("compile model...")
        self.model_train.compile(optimizer=self.optimizer, loss=loss_wrapper(alpha=alpha, beta=self.beta, gamma=self.gamma, cells=self.cells))
        #print(self.model_train.summary())

        newpath = self.model_path + run_name
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        tensorboard = self.tensorboard_path + run_name

        callbacks = [TensorBoard(log_dir=tensorboard, histogram_freq=0, write_graph=True, write_images=True)]
        # callbacks.append(LearningRateScheduler(lr_scheduler))
        #callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=self.patience, verbose=1, mode='auto'))
        #callbacks.append(ModelCheckpoint('./logs/' + run_name + '/weights.hdf5', monitor = 'val_loss', verbose = 1,
        #                                 save_best_only = False, save_weights_only = False, mode = 'auto', period = 1))
        callbacks.append(ModelCheckpoint('./logs/' + run_name + '/weights.{epoch:04d}.hdf5', monitor = 'val_loss', verbose = 1,
                                         save_best_only = False, save_weights_only = True, mode = 'auto', period = 1))

        input_list = [tensor_train[:,:,idx:idx+1] for idx in range(3*self.cov_num)]
        dummy_target = np.zeros(len(tensor_train[:, 0, 0]))

        self.model_train.fit(
            input_list,
            [dummy_target],
            batch_size=self.batch_size,
            epochs=self.epoch_max,
            callbacks=callbacks,
            #validation_split=self.valid_split,
            verbose=1)
        print('training complete!')

    def train_generator(self, run_name=None, data_file=None, alpha=None, test_set=None):
        """
        Default training scheme: generator training
        For each epoch fresh samples are drawn by triplet_generator
        After each epoch, the validation loss is monitored by an out-of-sample validation set
        Training stops if validation loss does not improve for patience epoch
        Additionally the customized Callback: MyCallback_test tests the model after each epoch
        regarding its re-identification strength (anchor-positive assignment) on a test set of your choice
        """

        if data_file is None:
            data_file = self.data_list

        if self.valid_split <= 0.0: raise Exception("for this type of training, a validation split is needed!")

        print("split training and validation set")
        train_set, valid_set = self.train_valid_split(data_file)
        print("length training set: ", len(train_set))
        print("length validation set: ", len(valid_set))

        # count valid anchor/positive users with at least 2 sequences
        count_valid = 0
        for var in valid_set:
            if len(var) > 1:
                count_valid += 1
            else:
                pass
        count_valid = count_valid * self.batches_per_user

        # count valid anchor/positive users with at least 2 sequences
        count_train = 0
        for var in train_set:
            if len(var) > 1:
                count_train += 1
            else:
                pass
        count_train = count_train * self.batches_per_user

        print("compile model...")
        self.model_train.compile(optimizer=self.optimizer,
                                 loss=loss_wrapper(alpha=alpha, beta=self.beta, gamma=self.gamma, cells=self.cells))
        print(self.model_train.summary())

        newpath = self.model_path + run_name
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        tensorboard = self.tensorboard_path + run_name

        callbacks = [TensorBoard(log_dir=tensorboard, histogram_freq=0, write_graph=True, write_images=True)]
        # callbacks.append(LearningRateScheduler(lr_scheduler))
        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=self.patience, verbose=1, mode='auto'))
        #callbacks.append(
        #    ModelCheckpoint('./logs/' + run_name + '/weights.{epoch:04d}.hdf5', monitor='val_loss', verbose=1,
        #                    save_best_only=False, save_weights_only=True, mode='auto', period=1))
        callbacks.append(MyCallback_test(self, test_set))

        print("valid anchor users in validation set (at least two sequences): ", count_valid)
        print("valid anchor users in training set (at least two sequences): ", count_train)

        self.model_train.fit_generator(
            self.triplet_generator_list_of_lists_cutoff(train_set),
            epochs=self.epoch_max,
            verbose=1,
            steps_per_epoch=int(count_train),  # *(self.triplets_per_user/batch_size)),
            callbacks=callbacks,
            # class_weight=NULL,
            max_queue_size=8,
            workers=8,
            initial_epoch=0,
            validation_data=self.triplet_generator_list_of_lists_cutoff(valid_set),
            validation_steps=int(count_valid),
        )
        print('training complete!')

    def train_file_generator(self, run_name=None, directory=None, alpha=None, number_of_user=None):

        self.model_train.compile(optimizer=self.optimizer, loss=loss_wrapper(alpha=alpha, beta=self.beta, gamma=self.gamma, cells=self.cells))
        print(self.model_train.summary())

        newpath = self.model_path + run_name
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        tensorboard = self.tensorboard_path + run_name

        callbacks = [TensorBoard(log_dir=tensorboard, histogram_freq=0, write_graph=True, write_images=True)]
        # callbacks.append(LearningRateScheduler(lr_scheduler))
        callbacks.append(
            EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=self.patience, verbose=1, mode='auto'))
        callbacks.append(ModelCheckpoint('./logs/' + run_name + '/weights.hdf5', monitor='val_loss', verbose=1,
                                         save_best_only=True, save_weights_only=False, mode='auto', period=1))
        # callbacks.append(MY_CALLBACK())

        self.model_train.fit_generator(
            self.file_generator(directory=directory, number_of_users_per_file=1000),
            epochs=self.epoch_max,
            verbose=1,
            steps_per_epoch=(number_of_user),
            # class_weight=NULL,
            workers=8,
            initial_epoch=0,
        )
        print('training complete!')


    def evaluate(self, triplet_tensor=None):
        """
        Evaluation study of triplet experiment (Paper Section 3.3)
        If the dist(anchor, positive) < dist(anchor, negative) count success
        """
        # reset counting statistic
        count_pos_manh = 0
        counter = 0

        # translate input array tensor into a list of arrays for each feature to feed it into model
        input_list = [triplet_tensor[:, :, idx:idx + 1] for idx in range(3 * self.cov_num)]

        prediction_anchor  = self.model_pred.predict(input_list[0:self.cov_num])
        prediction_positiv = self.model_pred.predict(input_list[self.cov_num:(2*self.cov_num)])
        prediction_negativ = self.model_pred.predict(input_list[(2*self.cov_num):(3*self.cov_num)])

        for idx_in in range(len(prediction_anchor)):
            dist_pos_manh = np.linalg.norm(prediction_anchor[idx_in] - prediction_positiv[idx_in], ord=1)
            dist_neg_manh = np.linalg.norm(prediction_anchor[idx_in] - prediction_negativ[idx_in], ord=1)
            if dist_pos_manh < dist_neg_manh:
                count_pos_manh += 1
            counter += 1

        self.score_L1 = count_pos_manh/counter
        print("score L1: ", self.score_L1)


#mymodel = tlrnn()
#tensor_training = datafromfile("data/triplet_tensor_top10k_itt_100k_restruct")

#tensor_pos = tensor_training.clip(min=0)

#mymodel.build()

#run_name= "test"
#alpha=10

#mymodel.train(run_name=run_name, tensor_train=tensor_pos, alpha=alpha)

tf.test.is_gpu_available( cuda_only=False, min_cuda_compute_capability=None )

print("finished")
