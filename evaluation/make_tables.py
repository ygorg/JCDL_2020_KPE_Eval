import os
import sys

import pandas as pd

from util import reference


def filter_ref(df, ref_dict):
    # Given a dictionary filled with dataset/reference_type pairs create a
    #  filter on df that filter out non chosen reference-types
    filter_ = None
    for dat, ref in ref_dict.items():
        tmp = ~((df['dataset'] == dat) & (df['reference'] != ref))
        if filter_ is None:
            filter_ = tmp
        else:
            filter_ &= tmp
    return filter_


abstract_dataset = ['Inspec', 'WWW', 'KP20k']  # 'KDD', 'TermiTH'
full_dataset = ['PubMed', 'ACM', 'SemEval-2010']  # 'CSTR', 'NUS', 'CiteULike-180'
news_dataset = ['DUC-2001', '500N-KPCrowd', 'KPTimes', 'NYTime']  # '110-PT-BN-KP', 'Wikinews'

all_datasets = full_dataset + abstract_dataset + news_dataset

models = [
    'FirstPhrases', 'TextRank', 'TfIdf', 'PositionRank', 'MultipartiteRank',
    'EmbedRank', 'Kea', 'CopyRNN', 'CopyCorrRNN', 'CopyRNN_News', 'CopyCorrRNN_News'
]
metrics = ['F@10', 'MAP']


if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'scores.csv'
    df = pd.read_csv(input_file)

    # Remove duplicates keeping the row with the highest 'time'
    tmp = df.groupby(list(set(df.columns) - set(['time', 'value'])))['time'].idxmax()
    df = df.loc[tmp]

    # Remove unwanted reference type (ex: for Inspec we will only use the 'uncontr' reference)
    ref_filter = filter_ref(df, reference)

    tmp = df[ref_filter][df['model'].isin(models)][df['metric'].isin(metrics)]

    # Create table from dataframe
    # Create the shape of the table, set the order of columns and lines
    # Note that we use `.max()` because we unsured with `filter_ref` that there was only one
    #  reference_type per dataset, the max is then computed on only one value

    """
    full = tmp[tmp['dataset'].isin(full_dataset)]
    full = full\
        .groupby(['model', 'dataset', 'metric']).max()['value']\
        .unstack(1).unstack().reindex(index=models)\
        .reindex(columns=full_dataset, level=0)\
        .apply(lambda x: round(x, 1)).fillna('n/a')
    print(full.to_latex())

    abstract = tmp[tmp['dataset'].isin(abstract_dataset)]
    abstract = abstract\
        .groupby(['model', 'dataset', 'metric']).max()['value']\
        .unstack(1).unstack().reindex(index=models)\
        .reindex(columns=abstract_dataset, level=0)\
        .apply(lambda x: round(x, 1)).fillna('n/a')
    print(abstract.to_latex())

    news = tmp[tmp['dataset'].isin(news_dataset)]
    news = news\
        .groupby(['model', 'dataset', 'metric']).max()['value']\
        .unstack(1).unstack().reindex(index=models)\
        .reindex(columns=news_dataset, level=0)\
        .apply(lambda x: round(x, 1)).fillna('n/a')
    print(news.to_latex())
    """

    all_ = tmp[tmp['dataset'].isin(all_datasets)]
    all_ = all_\
        .groupby(['model', 'dataset', 'metric']).max()['value']\
        .unstack(1).unstack().reindex(index=models)\
        .reindex(columns=all_datasets, level=0)\
        .apply(lambda x: round(x, 1)).fillna('n/a')
    print(all_.to_latex())

