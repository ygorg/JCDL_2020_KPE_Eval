#!/usr/bin/env python3

# Use case of this file: given a directory organized as follow:
# . / DATASET1 / DATASET1.METHOD1.info*.json
#                DATASET1.METHOD2.info*.json
#     DATASET2 / DATASET1.METHOD1.info*.json
#                DATASET1.METHOD3.info*.json
# the reference being located in : $PATH_AKE_DATASET/DATASET_/references/*
# and the original jsonl files in : $HOME/data/datasets/DATASET_.test.jsonl
# evaluate all method with every reference efficiently by cacheing every
# temporary files for reusing and storing all the result in a spreadsheet

import os
from glob import glob
import sys
import time
from itertools import product
import pandas as pd

from util import add_stem, rem_stem, get_metrics

"""
import pandas as pd
df = pd.read_csv('scores.csv')
# Removing duplicates:
tmp = df.groupby(list(set(df.columns) - set(['time', 'value'])))['time'].idxmax()
df = df.loc[tmp]

# Is there the same value for two different metrics ?
transpose = lambda x: list(zip(*x))
faulty_ids = []

piv_tab = df[list(set(df.columns) - set(['time']))].pivot_table(index=['model', 'dataset', 'r
eference'], columns='metric', values='value')
for m in ['F@10', 'P@10', 'R@10', 'F@5', 'P@5', 'R@5', 'MAP']:
    faulty_ids += transpose(piv_tab[piv_tab[m] == piv_tab[m+'_prs']].index.levels)
    faulty_ids += transpose(piv_tab[piv_tab[m] == piv_tab[m+'_abs']].index.levels)
set(faulty_ids)
"""


# TODO : make this script standalone or create some common functions for
#        evaluating


# If anything bad occurs save the cache
def signal_handler(sig=None, frame=None):
    global output
    global args
    global current_writing_file
    # If exiting and cacheing a file (like a .jsonl) delete the file so it is
    #  not incomplete
    if current_writing_file:
        os.remove(current_writing_file)
        current_writing_file = None
    logging.info('AtExit : Dumping to {}'.format(args.output_file))
    output.to_csv(args.output_file, index=False)
    sys.exit(0)


#####################
# Dealing with cache
#####################

# The date of modification is taken from the stemmed version and if it does not
#  exist from the not stemmed version
mod = lambda path: (os.path.getmtime(add_stem(path))
                    if os.path.isfile(add_stem(path))
                    else os.path.getmtime(path))


if __name__ == '__main__':
    import argparse
    import logging
    from tqdm import tqdm
    import atexit

    def arguments():
        parser = argparse.ArgumentParser(
            description='Evaluate all json file matching a glob pattern')
        parser.add_argument('dir', type=str,
                            help='Directory containing datasets/models_output')
        parser.add_argument('output_file', type=str,
                            help='Output file to update')
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='Print progress bar and logging.')
        # TODO : allow to compute only certain metrics (use case: not present
        #        and absent (because they take time))
        # parser.add_argument('-m', '--metrics', type=str,
        #                    help='Metrics to compute (Default: '
        #                         '[PRF]@[5,10][,_prs,_abs]')
        return parser.parse_args()

    args = arguments()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s:%(message)s',
            datefmt='%Y-%m-%d %I:%M:%S')

    if os.path.isfile(args.output_file):
        output = pd.read_csv(args.output_file, sep=',')
    else:
        output = pd.DataFrame(columns=[
            'model', 'dataset', 'reference', 'metric', 'value', 'time'])

    # Step 1: gathering all necessary information for every possible evaluation
    # path_to_dataset, path_to_reference, path_to_candidate
    # TODO: integrate 'split' variable
    # TODO: instead of iterating over DATASET/OUTPUTFILE, just walk to find .json files
    entries = {}
    for dataset_name in map(os.path.basename, glob(args.dir + '/*')):
        path_to_referenceS = glob(
            '{}/datasets/{}/references/test.*.json'.format(
                os.environ['PATH_AKE_DATASETS'], dataset_name))
        if not path_to_referenceS:
            logging.warning('No reference found for dataset {}'.format(
                dataset_name))
            continue

        path_to_candidateS = glob('{}/{}/{}.*.json'.format(
            args.dir, dataset_name, dataset_name))

        path_to_documents = os.environ['HOME'] + '/data/datasets/{}.test.jsonl'.format(
            dataset_name)
        if not os.path.exists(path_to_documents):
            logging.warning('Could not find jsonl file for {} dataset'.format(
                dataset_name))

        tmp = product(path_to_referenceS, path_to_candidateS)

        # Stem : we store the cannonical form of the file (non stemmed) and
        #        we'll add the stem at eval time (as we do when lowering string
        #        only for comparison)
        tmp = ((path_to_documents, rem_stem(r), rem_stem(c)) for r, c in tmp)
        tmp = set(tmp)

        entries[dataset_name] = tmp

    # Step 2: filter out already cached evaluations
    # Time : if old eval is older than dataset or reference or candidate then
    #        UPDATE

    # [P,R,F] @ [5,10] [,_abs,_prs]
    needed_metrics = ['MAP'] + [c + '@' + str(n) for c, n
                                in product('PRF', [5, 10])]
    needed_metrics = [a + b for a, b,
                      in product(needed_metrics, ['', '_abs', '_prs'])]
    needed_metrics = set(needed_metrics)

    entries = [c for e in entries.values() for c in e]

    new_entries = []
    for d, r, c in entries:
        documents_name = '.'.join(os.path.basename(d).split('.')[:-2])
        reference_name = '.'.join(os.path.basename(r).split('.')[1:-1])
        model_name = '.'.join(os.path.basename(c).split('.')[1:-1])
        # Search the cache for this entry
        computed_metrics = output[output['model'] == model_name][output['dataset'] == documents_name][output['reference'] == reference_name][['metric', 'time']]
        last_mod = max(mod(d), mod(r), mod(c))

        computed_metrics = computed_metrics[last_mod < computed_metrics['time']]['metric']
        computed_metrics = set(computed_metrics)
        # If some metrics need to be computed
        if needed_metrics - computed_metrics:
            new_entries.append((d, r, c, needed_metrics - computed_metrics))
    entries = new_entries

    # Step 3: perform the evaluation

    # By now if a problem occur save the computed metrics
    atexit.register(signal_handler)
    current_writing_file = None

    # Deal with cacheing things
    for d, r, c, m in tqdm(entries, disable=not args.verbose):
        documents_name = '.'.join(os.path.basename(d).split('.')[:-2])
        reference_name = '.'.join(os.path.basename(r).split('.')[1:-1])
        model_name = '.'.join(os.path.basename(c).split('.')[1:-1])

        logging.info(documents_name + reference_name + model_name)

        try:
            metrics = get_metrics(d, r, c, needed_metrics=m)
        except:
            continue

        for m_name, m_val in metrics:
            # TODO: Remove old entry if existing
            output = output.append({
                'model': model_name, 'dataset': documents_name,
                'reference': reference_name, 'metric': m_name,
                'value': round(m_val, 4), 'time': int(time.time())
            }, ignore_index=True)

    logging.info('Dumping to {}'.format(args.output_file))
    output.to_csv(args.output_file, index=False)
