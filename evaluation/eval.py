#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Use case of this file : compute PRF given a candidate file and a reference
#  file. Can also filter keyphrases to evaluate only present/absent keyphrases.

# TODO: Replace measures computation by util.eval_measures


from __future__ import division

if __name__ == '__main__':
    import os
    import json
    import logging
    import argparse
    from tqdm import tqdm
    from util import compute_rel, compute_prf, compute_averagep
    from util import stem_data, stemmed, absent, present

    def arguments():
        parser = argparse.ArgumentParser(
            description='Keyphrase extraction performance evaluation script.')
        parser.add_argument(
            '-i', '--input', required=True, type=argparse.FileType('r'),
            help='path to file containing the candidate keyphrases')
        parser.add_argument(
            '-r', '--reference', required=True, type=argparse.FileType('r'),
            help='path to file containing the reference keyphrases')
        parser.add_argument(
            '-n', '--nbest', default=10, type=int,
            help='top-n candidate keyphrases to assess')
        parser.add_argument(
            '-o', '--output', type=str,
            help='output file for outputting results')
        parser.add_argument(
            '--present', type=str,
            help='Path to original data. Evaluate on present keyphrases only '
                 '(filter input)')
        parser.add_argument(
            '--absent', type=str,
            help='Path to original data. Evaluate on absent keyphrases only'
                 '(filter input)')
        parser.add_argument(
            '--no-filter-ref', action='store_true',
            help='If absent or present, do not filter the reference. This will '
                 'evaluate the prs/abs keyphrases on the whole reference.')
        parser.add_argument(
            '-v', '--verbose', action='store_true',
            help='Print logging and progress bar')
        return parser.parse_args()

    args = arguments()
    args.input = args.input.name
    args.reference = args.reference.name

    # Deal with --verbose argument
    def progress(x, **kwargs):
        return x
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        progress = tqdm

    # Preprocess and cache, keyphrases and data

    if args.present or args.absent:
        if args.present and args.absent:
            raise ArgumentError('Cannot filter absent and present')
        logging.info('Loading and stemming original data, filtering candidates'
                     + ('' if args.no_filter_ref else ' and reference'))
        if args.present:
            file_path = stem_data(args.present)
            filt_func = present
        elif args.absent:
            file_path = stem_data(args.absent)
            filt_func = absent

        args.input = filt_func(stemmed(args.input), file_path)
        if not args.no_filter_ref:
            args.reference = filt_func(stemmed(args.reference), file_path)
    # s_c, s_r = set(candidates), set(references)
    # if len(s_c & s_r) != len(s_c | s_r):
    #    logging.warning('#cand : {}, #ref : {}'.format(
    #        len(candidates), len(references)))
    #    logging.warning('{} documents will be evaluated on {} total '
    #                    'documents'.format(len(s_c & s_r), len(s_c | s_r)))

    logging.info('Loading candidate keyphrases from {}'.format(args.input))
    try:
        with open(args.input) as f:
            candidates = json.load(f)
    except Exception:
        logging.error('Error loading {}'.format(args.input))
        exit(-1)

    logging.info('Loading reference keyphrases from {}'.format(
        args.reference))
    with open(args.reference) as f:
        references = json.load(f)

    logging.info('Evaluating keyphrase extraction performance {}'.format(
        args.input))
    logging.info('Number of files: {}'.format(len(candidates)))

    precisions = []
    recalls = []
    f_scores = []
    average_p = []

    if args.output:
        output_file = open(args.output, 'w')

    for doc_id in progress(candidates):
        if doc_id not in references:
            continue
        rel = compute_rel(candidates[doc_id], references[doc_id])

        max_len = min(len(candidates[doc_id]), args.nbest)

        p, r, f = compute_prf(rel[:max_len], references[doc_id])
        ap = compute_averagep(rel, references[doc_id])

        precisions.append(p)
        recalls.append(r)
        f_scores.append(f)
        average_p.append(ap)

        if args.output:
            output_file.write(
                "{}\t{}\t{}\t{}\t{}\n".format(doc_id, p, r, f, ap)
            )

    P = sum(precisions) / len(precisions) * 100.0
    R = sum(recalls) / len(recalls) * 100.0
    F = sum(f_scores) / len(f_scores) * 100.0
    MAP = sum(average_p) / len(average_p) * 100.0

    if args.output:
        output_file.close()

    print("| {1:5.2f} | {2:5.2f} | {3:5.2f} | {4:5.2f} | {0:2d} | {5} |".format(
          args.nbest, P, R, F, MAP, os.path.basename(args.input)))
