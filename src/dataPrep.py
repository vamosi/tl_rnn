"""
TL-RNN Triplet Loss Recurrent Neural Networks - a sequence embedder for discrete time-series data-sets
for segmentation or re-identification of human behavior

Author: Stefan Vamosi
"""

import pandas as pd
import os
import numpy as np
from datetime import datetime
from util import *
import random
from config import BATCHES_PER_USER, BATCH_SIZE, COV_NUMBER, SEQUENCE_LEN_MAX, COVARIATES_ALL, COVARIATES_TO_TRANSLATE, \
                        MAX_CARDINALITY, TAKE_LAST_ACTIONS


class dataprep():
    """
    dataprep class to prepare dataframe to readable tensor of numbers for effective sampling in model class
    input datafram has to be column-wise dataset with id-column defined with id_name in config file like:
    "id_name" | feature 1 | feature 2 | ....
    232324    | clickA    | value10   | ....
    ......
    """
    def __init__(self, dataframe = None, id_name = None, sequence_id = "week_number"):
        """
        reads in dataframe
        determine parameters and print them
        run initial scripts like determine ranke-based-encoding
        vectorize encodings for faster pre-processing
        params:
            dataframe: sequential dataframe to work on
            id_name:        column name of the user_id or something that identifies individuals
            sequence_id:    column name that contains a sequence identifier,
                            could be a timeframe like week_number or a session_id
        """
        self.id_name = id_name
        self.dataframe = dataframe.fillna(0)
        self.sequence_id = sequence_id
        self.covariates_all = COVARIATES_ALL
        self.covariates_to_translate = COVARIATES_TO_TRANSLATE
        self.cardinalities = {}
        self.triplets_per_user = BATCHES_PER_USER*BATCH_SIZE
        self.sequence_len_max = SEQUENCE_LEN_MAX
        self.cov_number = COV_NUMBER
        self.max_card = MAX_CARDINALITY
        self.triplet_tensor = None
        self.domains_max = None
        self.ids = None
        self.flag_overwrite = False
        self.overwritten = []
        self.data_list = []
        self.machine_id_list = []
        self.ranks = self.most_common_func(self.dataframe)
        self.vectorized_convert = np.vectorize(self.convert_cov)
        self.take_last_actions = TAKE_LAST_ACTIONS
        self.covariates_cat_indx = []
        i = 0
        while (i < len(self.covariates_all)): ## find out indexes of categorical covariates
            if (self.covariates_to_translate.count(self.covariates_all[i]) > 0):
                self.covariates_cat_indx.append(i)
            i += 1
        print("DATAFRAME COLUMNS ", list(self.dataframe.columns))
        print("SEQ_LEN_MAX" + '\033[95m', self.sequence_len_max)
        print("TOP 20 Categories of all Covariates", self.ranks[:20])
        print("COVARIATES_ALL" + '\033[95m', self.covariates_all)
        print("SAMPLES PER USER", self.triplets_per_user)
        print("SEQUENCE LENGTH MAX", self.sequence_len_max)

    def most_common_func(self, dataframe):
        """
        determine the cardinality and ranks of all categorical features for encoding
        """
        rank_list = []
        for cov in self.covariates_to_translate:
            df_in = dataframe.groupby(cov).size().to_frame('size')
            df_in['rank'] = df_in.rank(ascending=0)
            df_in = df_in.sort_values(by=['rank'], ascending=True)
            one_list = [cov for cov in df_in.index[:len(df_in)]]
            rank_list.append(one_list)
        return rank_list

    def convert_cov(self, cov, cov_index):
        """
        assing rank-based encoding to signals, if this signal is new or higher than max_card -> assign "(other)" category

        input:  cov = categorie of a feature (covariate)
                cov_index = ith covariate (column)

        return: ranked based encoding
        """
        if (cov in self.ranks[cov_index]) and (self.ranks[cov_index].index(cov) <= self.max_card):
            out = self.ranks[cov_index].index(cov)
        else:
            out = self.ranks[cov_index].index("(other)")
            self.flag_overwrite = True
            self.overwritten.append(cov)
        return out

    def preprocessor(self):
        """
        move data from original dataframe into lists of arrays,
        encode them properly: categories with a ranked based encoding
        one list entry per user array, with [timeframe index (e.g. nth week), seq_len(e.g. 1000), feature]
        time order conserved (t-1,t0,t+1)

        params: only members coming from dataprep init

        input:  columnwise dataframe read by dataframe() init

        return: list with each element being a user nested with another list of single array sequences:
                [user][array[timestep, features]]
        """
        print("dataframe -> tensor ...")
        self.ids = pd.unique(self.dataframe[self.id_name])  # include all machine_ids
        for indx, machine_id in enumerate(self.ids):
            if indx%1000 == 0:
                print("--------------------------------------------------")
                print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                print('{} of {} users'.format(indx, len(self.ids)), "\r")
            # extract the id's block
            df_id = self.dataframe.loc[self.dataframe[self.id_name] == machine_id]
            # calculate sequence along following column feature, e.g. "week_number", or "month"
            first_dim = pd.unique(df_id[self.sequence_id])
            # for each user one tensor list, where the arrays are stored with their's sequence length
            tensor_list = []
            # domain as number
            for i, t in enumerate(first_dim):
                # extract one sequence from dataframe according to sequence length (week or day as specified in self.sequence_id)
                df_id_t = df_id.loc[df_id[self.sequence_id] == t]
                # initialize array of the week
                week_array = np.zeros((len(df_id_t), len(self.covariates_all)), dtype='int32')
                # cap block length with sequence_len_max:
                for col in self.covariates_all:
                    # covariates_need to be translated according ranked-based-encoding, always reserve 0 for no data (end of observation)
                    if col in self.covariates_to_translate:
                        week_array[:, self.covariates_all.index(col)] = \
                            np.asarray(self.vectorized_convert(df_id_t.as_matrix(columns=[col]), self.covariates_to_translate.index(col)) + 1).reshape(-1)
                    # loop through all other features and write them directly to array
                    else:
                        week_array[:, self.covariates_all.index(col)] = np.asarray(df_id_t.as_matrix(columns=[col]) + 1).reshape(-1)
                tensor_list.append(week_array)
            self.data_list.append(tensor_list)
            self.machine_id_list.append(machine_id)
        # give alarm if a domain is not listed
        if self.flag_overwrite:
            print("domains ", self.overwritten, "overwritten with ", "(other)")
        print("tensor builder finished")

    def build_cardinality(self, data_list = None):
        """
        evaluate the cardinality (max value) of each feature and add +1 to account for no information
        put this object (covariates) into the tl_rnn class like: tlrnn(cells=128, covariates=dataobj.covariates)

        params: None

        input: data_list coming from dataprep.preprocessor

        return: self.covariates (put it into model)
        """
        if data_list:
            data_list = data_list  # self.triplets_per_user
        else:
            data_list = self.data_list

        max_array = np.zeros((len(self.cov_number)), dtype='int32')

        for user_list in data_list:
            for seq in user_list:
                for feature in self.cov_number:
                    if max(seq[:, feature]) > max_array[feature]:
                        max_array[feature] = max(seq[:, feature])
        max_array = max_array + 1

        self.cardinalities = dict(zip(self.covariates_all, max_array.tolist()))
        print(self.cardinalities)

    def sampler(self, input_list=None, directory=None, triplets_per_user = None, users_per_file="all", savefile=False):
        """
        sample triplets directly from dataframe with a fixed time-span and a capped sequence length

        params: directory: If save_file then this directory says where
                users_per_file: here you can set how many users you want to sample per file, put in "all" for all in one file
                triplets_per_user:  overwrite here, if you don't like the batch_size*batches_per_user (for epoch eval e.g.)
                                    consider batch_size to avoid remainder!
                take_last_actions [default=True]: when sequence too long crop it so that last actions are conserved

        input:  data_list coming from dataprep.preprocessor()
                If datalist from external, put it in here, otherwise data_list from object is taken

        return: verwrite self.triplet_tensor
        """

        if triplets_per_user is None:
            triplets_per_user = self.triplets_per_user

        print("triplets per user: ", triplets_per_user)
        if input_list:
            data_list = input_list
        else:
            data_list = self.data_list

        user_list = list(range(0, len(data_list)))

        # allow for one datafile output (all users in one)
        if users_per_file == "all":
            # prepare triplet tensor to fed into model training
            users_per_file = len(data_list)

        # prepare triplet tensor to fed into model training
        # if more than one file, prepare the tensor accordingly
        triplet_tensor = np.zeros(((users_per_file * self.triplets_per_user),
                                            self.sequence_len_max, 3 * len(self.covariates_all)), dtype='int32')

        # index for main triplet tensor over all triplets and users
        indx_in = 0
        for indx, user in enumerate(data_list):
            if indx % 1000 == 0:
                print('{} of {} users'.format(indx, len(data_list)), "\r")
            for _ in range(self.triplets_per_user):
                if len(user) > 1:
                    anchor, positive = random.sample(list(user), 2)

                    user_list_pop = user_list.copy()
                    user_list_pop.pop(indx)
                    neg_indx = random.choice(user_list_pop)
                    negative = random.sample(list(data_list[neg_indx]), 1)[0]

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

                    triplet_tensor[indx_in, (self.sequence_len_max - len(anchor[:,0])):, 0:len(self.covariates_all)] = anchor
                    triplet_tensor[indx_in, (self.sequence_len_max - len(positive[:,0])):, len(self.covariates_all):2 * len(self.covariates_all)] = positive
                    triplet_tensor[indx_in, (self.sequence_len_max - len(negative[:,0])):, 2 * len(self.covariates_all):3 * len(self.covariates_all)] = negative

                    indx_in += 1

            self.triplet_tensor = triplet_tensor

            if savefile:
                if ((indx > 0) and ((indx+1) % users_per_file) == 0) or (indx == (len(data_list) - 1)):
                    print('Save data chunk'.format(indx, " of ", len(data_list)), "\r")
                    pickle.dump(self.triplet_tensor, open(directory + "triplet_file_usergroup" + str(indx), 'wb'),
                                protocol=4)
                    triplet_tensor = np.zeros(((users_per_file * self.triplets_per_user),
                                               self.sequence_len_max, 3 * len(self.covariates_all)), dtype='int32')
                    indx_in = 0

