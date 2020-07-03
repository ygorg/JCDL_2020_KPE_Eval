#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import

import os
import sys
import json
import gzip
import glob
import copy
import bisect
import codecs
import string
import logging
import argparse

from collections import defaultdict

import pke
from nltk.corpus import stopwords
from pke.base import ISO_to_language


def prune_df_counts(df_counts, candidates):
    """Prune the DF counts from the given candidates."""

    # copy to DF counts
    df = copy.deepcopy(df_counts)

    # remove one document from the count
    df['--NB_DOC--'] -= 1

    # then remove the counts of the candidates
    for candidate in candidates:
        if candidate not in df:
            continue
        if df[candidate] == 1:
            del df[candidate]
        else:
            df[candidate] -= 1

    return df


def load_pairwise_similarities(path):
    """Load the pairwise similarities for ExpandRank."""

    pairwise_sim = defaultdict(list)
    with gzip.open(path, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            cols = line.decode('utf-8').strip().split()
            bisect.insort(pairwise_sim[cols[0]], (float(cols[2]), cols[1]))
            bisect.insort(pairwise_sim[cols[1]], (float(cols[2]), cols[0]))
    return pairwise_sim


def load_cluster_similarities(path):
    # Load the similarities for CollabRank.
    cluster_sim = defaultdict(list)
    with gzip.open(path, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            cols = line.decode('utf-8').strip().split("\t")
            cluster_sim[cols[0]] = cols[1:]
    return cluster_sim


parser = argparse.ArgumentParser(description='Run AKE experiment')

parser.add_argument(
    '-p', '--parameter_file', required=True,
    help='Path to the parameter file')

parser.add_argument(
    '-v', '--verbose', action='store_true',
    help='Verbose mode')

args = parser.parse_args()

if not os.environ['PATH_AKE_DATASETS']:
    logging.error('Please set environment variable `PATH_AKE_DATASETS` using'
                  '`export PATH_AKE_DATASETS="PATH/TO/ake-datasets"`')
    logging.error('Exiting...')
    exit()

# activate verbose mode
if args.verbose:
    logging.basicConfig(level=logging.DEBUG)

# load parameters
logging.info("loading parameters from {}".format(args.parameter_file))
with open(args.parameter_file) as f:
    params = json.load(f)
    params["path"] = os.path.join(os.environ['PATH_AKE_DATASETS'],
                                  params["path"])
    params["reference"] = os.path.join(os.environ['PATH_AKE_DATASETS'],
                                       params["reference"])

###############################################################################
# GLOBAL VARIABLES INITIALIZATION
###############################################################################

# initialize stoplist
punctuations = list(string.punctuation)
punctuations += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
stoplist = stopwords.words(ISO_to_language[params["language"]]) + punctuations

# initialize path to the test files
path_to_test = "{}/test/".format(params["path"])

# initialize path to the train file
only_test = False
path_to_train = "{}/train/".format(params["path"])
if not os.path.isdir(path_to_train):
    logging.info("no training data available, switching to 'only test' mode")
    path_to_train = path_to_test
    only_test = True

# initialize the split for training/computing stats
split_for_training = "train"
if only_test:
    split_for_training = "test"

# initialize the path to the DF file
path_to_df_file = "{}/{}.{}.df.tsv.gz".format(
    params["output"],
    params["dataset_identifier"],
    split_for_training)

# initialize the path to the LDA distributions file
path_to_lda_file = "{}/{}.{}.lda-{}.pickle.gz".format(
    params["output"],
    params["dataset_identifier"],
    split_for_training,
    params["n_topics"])

# initialize the path to the pairwise similarities
path_to_pairwise_file = "{}/{}.{}.pairwise.gz".format(
    params["output"],
    params["dataset_identifier"],
    split_for_training)

# initialize the path to the Kea model
path_to_kea_file = "{}/{}.{}.kea.pickle".format(
    params["output"],
    params["dataset_identifier"],
    split_for_training)

# initialize the path to the leave-one-out Kea models
path_to_leave_one_out_models = params["output"] + "/loo/"

# DF counts
df = {}
pairwise = {}

###############################################################################

###############################################################################
# PRE-COMPUTING WEIGHTS/STATS
###############################################################################

# pre-compute DF weights if needed
need_df = any(model in ['KPMiner', 'Wingnus', 'TfIdf', 'Kea']
              for model in params['models'])
if need_df and not os.path.isfile(path_to_df_file):
    logging.info("computing DF weights from {}".format(params["path"]))
    pke.compute_document_frequency(input_dir=path_to_train,
                                   output_file=path_to_df_file,
                                   extension=params["extension"],
                                   language=params["language"],
                                   normalization=params["normalization"],
                                   stoplist=punctuations,
                                   delimiter='\t',
                                   n=5)

# pre-compute LDA distributions if needed
need_lda = any(model in ['TopicalPageRank'] for model in params['models'])
if need_lda and not os.path.isfile(path_to_lda_file):
    logging.info("computing LDA distributions from {}".format(params["path"]))
    pke.compute_lda_model(input_dir=path_to_train,
                          output_file=path_to_lda_file,
                          n_topics=params["n_topics"],
                          extension=params["extension"],
                          language=params["language"],
                          normalization=params["normalization"])


# pre-compute pairwise similarities if needed
need_pairwise = any(model in ['ExpandRank'] for model in params['models'])
if need_pairwise and not os.path.isfile(path_to_pairwise_file):
    logging.info("computing pairwise similarities in {}".format(
        params["path"]))

    logging.info("loading DF counts from {}".format(path_to_df_file))
    df_counts = pke.load_document_frequency_file(input_file=path_to_df_file)

    pke.compute_pairwise_similarity_matrix(
        input_dir=path_to_test,
        output_file=path_to_pairwise_file,
        collection_dir=path_to_train,
        df=df_counts,
        extension=params["extension"],
        language=params["language"],
        normalization=params["normalization"],
        stoplist=stoplist)

###############################################################################

###############################################################################
# TRAINING SUPERVISED MODEL
###############################################################################

if not only_test:
    # Training a supervised Kea model
    if not os.path.isfile(path_to_kea_file):

        logging.info("Training supervised model {}".format(path_to_kea_file))

        logging.info("loading DF counts from {}".format(path_to_df_file))
        df_counts = pke.load_document_frequency_file(
            input_file=path_to_df_file)

        pke.train_supervised_model(input_dir=path_to_train,
                                   reference_file=params["reference"],
                                   model_file=path_to_kea_file,
                                   extension=params["extension"],
                                   language=params["language"],
                                   normalization=params["normalization"],
                                   df=df_counts,
                                   model=pke.supervised.Kea())

else:
    # No training set is available
    if not os.path.isdir(path_to_leave_one_out_models):
        os.makedirs(path_to_leave_one_out_models)

        logging.info("Training LOO models {}".format(
            path_to_leave_one_out_models))

        logging.info("loading DF counts from {}".format(path_to_df_file))
        df_counts = pke.load_document_frequency_file(
            input_file=path_to_df_file)

        path_to_LOO_kea_file = "{}/{}".format(
            path_to_leave_one_out_models,
            params["dataset_identifier"])

        pke.train_supervised_model(input_dir=path_to_train,
                                   reference_file=params["reference"],
                                   model_file=path_to_LOO_kea_file,
                                   extension=params["extension"],
                                   language=params["language"],
                                   normalization=params["normalization"],
                                   df=df_counts,
                                   model=pke.supervised.Kea(),
                                   leave_one_out=True)


###############################################################################


###############################################################################
# KEYPHRASE EXTRACTION
###############################################################################

# loop through the models
for model in params["models"]:

    output_file = "{}/{}.{}.json".format(params["output"],
                                         params["dataset_identifier"],
                                         model)

    stemmed_output_file = "{}/{}.{}.stem.json".format(
        params["output"],
        params["dataset_identifier"],
        model)

    if not os.path.isfile(stemmed_output_file):
        logging.info("running [{}]".format(stemmed_output_file))

        # container for keyphrases
        keyphrases = {}
        stemmed_keyphrases = {}

        # switch for one model Kea
        one_model_kea = False
        if model == "KeaOneModel":
            model = "Kea"
            only_test = False
            one_model_kea = True
            path_to_df_file = params["kea_df_weights"]

        # get class from module
        class_ = getattr(pke.unsupervised, model, None)

        if not class_:
            class_ = getattr(pke.supervised, model, None)
            if not class_:
                logging.error('[{}] is not a valid pke model'.format(model))
                sys.exit(0)

        # test if collection of documents is a zip
        # with zipfile.ZipFile(sys.argv[2], 'r') as f:
        if path_to_test.endswith('.zip'):
            logging.warning("{} id a zip file".format(path_to_test))
            continue

        # loop through the documents
        for input_file in glob.iglob(os.path.join(path_to_test,
                                     '*.' + params['extension'])):

            # get the document identifier
            file_id = input_file.split("/")[-1][:-4]

            logging.info("extracting keyphrases from [{}]".format(file_id))

            # initialize the ake model
            extractor = class_()

            # read the document
            extractor.load_document(input=input_file,
                                    language=params["language"],
                                    normalization=params["normalization"])

            # extract the keyphrase candidates
            extractor.grammar_selection(grammar=params["grammar"])

            # filter candidates containing stopwords or punctuation marks
            extractor.candidate_filtering(stoplist=stoplist,
                                          minimum_length=3,
                                          minimum_word_size=2,
                                          valid_punctuation_marks='-',
                                          maximum_word_number=5,
                                          only_alphanum=True)

            # rank candidates
            if model in ['TfIdf', 'KPMiner']:
                if not df:
                    logging.info("loading DF weights from {}".format(
                        params["path"]))
                    df = pke.load_document_frequency_file(
                        input_file=path_to_df_file)

                if only_test:
                    logging.info("pruning DF counts")
                    pruned_df = prune_df_counts(df, list(extractor.candidates))
                    extractor.candidate_weighting(df=pruned_df)

                else:
                    extractor.candidate_weighting(df=df)

            elif model in ['PositionRank', 'TextRank']:
                extractor.candidate_weighting(pos=params["pos"])

            elif model == "TopicalPageRank":
                extractor.candidate_weighting(pos=params["pos"],
                                              lda_model=path_to_lda_file)

            elif model == "TopicCoRank":
                if only_test:
                    extractor.candidate_weighting(
                        input_file=params["reference"],
                        excluded_file=file_id,
                        lambda_t=0.5,
                        lambda_k=0.5)
                else:
                    extractor.candidate_weighting(
                        input_file=params["reference"],
                        lambda_t=0.5,
                        lambda_k=0.5)

            elif model == "ExpandRank":
                if not pairwise:
                    pairwise = load_pairwise_similarities(
                        path=path_to_pairwise_file)

                expanded_documents = [(v, u) for u, v in pairwise[input_file][-params["n_expanded"]:]]
                extractor.candidate_weighting(
                    pos=params["pos"],
                    expanded_documents=expanded_documents)
            elif model == "CollabRank":
                if not cluster_sim:
                    cluster_sim = load_cluster_similarities(
                        path=path_to_cluster)

                collab_documents = [(data.split(",")[0], data.split(",")[1])
                                    for data in cluster_sim[input_file]]

                extractor.candidate_weighting(
                    collab_documents=collab_documents,
                    pos=params["pos"])
            elif model in ['Kea']:
                if not df:
                    logging.info("loading DF weights from {}".format(params["path"]))
                    df = pke.load_document_frequency_file(
                        input_file=path_to_df_file)

                if only_test:
                    logging.info("pruning DF counts")
                    pruned_df = prune_df_counts(df, list(extractor.candidates))

                    # initialize the path to the LOO Kea model
                    path_to_LOO_kea_file = "{}/{}.{}.pickle".format(
                        path_to_leave_one_out_models,
                        params["dataset_identifier"],
                        file_id)

                    if one_model_kea:
                        path_to_LOO_kea_file = params["kea_model"]
                        logging.info(
                            "switching to One Kea Model from {}".format(
                                params["kea_model"]))

                    extractor.candidate_weighting(
                        model_file=path_to_LOO_kea_file,
                        df=pruned_df)

                else:
                    if one_model_kea:
                        path_to_kea_file = params["kea_model"]
                        logging.info(
                            "switching to One Kea Model from {}".format(
                                params["kea_model"]))

                    extractor.candidate_weighting(
                        model_file=path_to_kea_file,
                        df=df)

            else:
                extractor.candidate_weighting()

            # pour the nbest in the containers
            kps = extractor.get_n_best(n=params["nbest"], stemming=False)
            keyphrases[file_id] = [[u] for (u, v) in kps]

            s_kps = extractor.get_n_best(n=params["nbest"], stemming=True)
            stemmed_keyphrases[file_id] = [[u] for (u, v) in s_kps]

        logging.info('writting ouput in {}'.format(output_file))
        with codecs.open(output_file, 'w', 'utf-8') as o:
            json.dump(keyphrases, o, sort_keys=True, indent=4)

        logging.info('writting ouput in {}'.format(stemmed_output_file))
        with codecs.open(stemmed_output_file, 'w', 'utf-8') as o:
            json.dump(stemmed_keyphrases, o, sort_keys=True, indent=4)

###############################################################################
