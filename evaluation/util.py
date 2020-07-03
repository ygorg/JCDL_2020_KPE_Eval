import os
import json
import logging

from tqdm import tqdm
from nltk import word_tokenize
from nltk.stem import PorterStemmer

# Default reference type for a corpus
reference = {
    'CSTR': 'author',
    'NUS': 'combined',
    'PubMed': 'author',
    'ACM': 'author',
    'ACM-abstract': 'author',
    'Citeulike-180': 'reader',
    'SemEval-2010': 'combined',
    'SemEval-2010-abstract': 'combined',
    'Inspec': 'uncontr',
    'KDD': 'author',
    'WWW': 'author',
    'TermITH': 'indexer',
    'KP20k': 'author',
    'DUC-2001': 'reader',
    '110-PT-BN-KP': 'reader',
    '500N-KPCrowd': 'reader',
    'TermITH-Eval': 'indexer',
    'WikinewsKeyphrase': 'reader',
    'KPTimes': 'editor',
    'NYTime': 'editor'
}


def compute_rel(candidate, reference):
    """Computes matches between candidate and reference.

    Keyphrases may have variants thus the List[List[str]] construct.
    :param candidate: List of candidate keyphrases.
    :type candidate: List[List[str]]
    :param reference: List of reference keyphrases.
    :type reference: List[List[str]]
    :returns: A list of len(candidate) lists containing the ids of matched
              references.
    :rtype: List[List[int]]
    """
    rel = []
    for k in candidate:
        # rel_k contains the references matching by candidate at rank k
        # This is a list because there can be multiple occurence of the same
        # reference (after stemming for ex)
        rel_k = []

        # For each reference check whether its variants intersect
        # with candidate at rank K variants

        for j, gold_variants in enumerate(reference):
            if set(gold_variants).intersection(k):
                rel_k.append(j)
        rel.append(rel_k)
    return rel


def compute_prf(rel, reference):
    """Compute precision, recall and f-score.

    If no reference or no match the result will be 0.
    :param rel: Output of `compute_rel`. A list of lists containing the ids of
                matched references.
    :type rel: List[List[int]]
    :param reference: List of reference keyphrases. Only the length is used.
    :type reference: List[List[str]]
    :returns: Precision, Recall and F1-Score
    :rtype: Tuple[float, float, float]
    """
    nb_matches = len(set(sum(rel, [])))

    p, r, f, = 0, 0, 0
    if rel:
        p = float(nb_matches) / len(rel)
    if reference:
        r = float(nb_matches) / len(reference)

    if p > 0 or r > 0:
        f = (2.0 * p * r) / (p + r)

    return p, r, f


def compute_averagep(rel, reference):
    """Computes Average Precision.

    If no reference the output is 0.
    :param rel: Output of `compute_rel`. A list of lists containing the ids of
                matched references.
    :type rel: List[List[int]]
    :param reference: List of reference keyphrases. Only the length is used.
    :type reference: List[List[str]]
    :returns: Average precision.
    :rtype: float
    """
    if not reference:
        return 0
    # If the candidate at rank i is valid
    # Compute the number of reference matched by candidates from rank 1 to i
    p_at_k = [len(set(sum(rel[:i], []))) / i
              for i in range(1, len(rel) + 1)
              if rel[i - 1]]
    return sum(p_at_k) / len(reference)


def eval_measures(path_to_model, path_to_reference, n=None):
    """Compute P, R, F, MAP at n.

    [description]
    :param path_to_model: path to candidate keyphrases (json format)
    :type path_to_model: str
    :param path_to_reference: path to reference keyphrases (json format)
    :type path_to_reference: str
    :param n: top-n candidate keyphrase to consider, defaults to 10
    :type n: int, optional
    :returns: A dictionary of measures, for every `n` corresponding P@N, R@N and
              F@N are available.
    :rtype: Dict[str, float]
    """
    # TODO: This function should write a file 'path_to_model.reftype.n.scores'

    if n is None:
        n = 10
    if type(n) is int:
        n = [n]

    with open(path_to_model) as f:
        candidates = json.load(f)
        # candidates : dict{doc_id -> list(list(keyphrase))}
    with open(path_to_reference) as f:
        references = json.load(f)
        # references : dict{doc_id -> list(list(keyphrase))}

    measures = [m + '@' + i for m in 'PRF' for i in map(str, n)] + ['MAP']
    measures = {k: [] for k in measures}

    for doc_id in candidates:
        if doc_id not in references:
            continue
        rel = compute_rel(candidates[doc_id], references[doc_id])

        ap = compute_averagep(rel, references[doc_id])
        measures['MAP'].append(ap)

        for i in n:
            max_len = min(len(candidates[doc_id]), i)

            p, r, f = compute_prf(rel[:max_len], references[doc_id])
            measures['P@' + str(i)].append(p)
            measures['R@' + str(i)].append(r)
            measures['F@' + str(i)].append(f)

    measures = {k: sum(v) / len(v) * 100. for k, v in measures.items()}

    return measures


# Compute the needed metrics for the set of provided files
def get_metrics(path_to_documents, path_to_reference,
                path_to_model, needed_metrics=[]):
    # TODO : how to make this function more generalizable and allow to introduce
    #        other metrics
    # TODO : remove hard_coded n=[5, 10]
    path_to_model = stemmed(path_to_model)
    path_to_reference = stemmed(path_to_reference)

    tmp_metrics = {}

    try:

        if any('_abs' not in m and '_prs' not in m for m in needed_metrics):
            logging.debug('Vanilla measures')
            tmp_metrics = eval_measures(
                path_to_model,
                path_to_reference,
                n=[5, 10])
            for n, v in tmp_metrics.items():
                yield n, v

        path_to_documents = stem_data(path_to_documents)

        if any('_abs' in m for m in needed_metrics):
            logging.debug('Absent measures')
            tmp_metrics = eval_measures(
                absent(path_to_model, path_to_documents),
                absent(path_to_reference, path_to_documents),
                n=[5, 10])
            tmp_metrics = {k + '_abs': v for k, v in tmp_metrics.items()}
            for n, v in tmp_metrics.items():
                yield n, v

        if any('_prs' in m for m in needed_metrics):
            logging.debug('Present measures')
            tmp_metrics = eval_measures(
                present(path_to_model, path_to_documents),
                present(path_to_reference, path_to_documents),
                n=[5, 10])
            tmp_metrics = {k + '_prs': v for k, v in tmp_metrics.items()}
            for n, v in tmp_metrics.items():
                yield n, v
    except Exception as e:
        documents_name = '.'.join(os.path.basename(path_to_documents).split('.')[:-2])
        reference_name = '.'.join(os.path.basename(path_to_reference).split('.')[1:-1])
        model_name = '.'.join(os.path.basename(path_to_model).split('.')[1:-1])
        logging.error('Error ({}) with {} {} {}'.format(
            type(e), documents_name, reference_name, model_name))
        print(e)
        logging.error('Files involved:')
        logging.error(path_to_documents)
        logging.error(path_to_reference)
        logging.error(path_to_model)
        return []


def get_ref(file_name, split='test'):
    corpus_name = os.path.basename(file_name).split('.')[0]
    ref_dir = os.path.join(
        os.environ['PATH_AKE_DATASETS'], 'datasets',
        corpus_name, 'references')

    stemmed = 'stem' in file_name
    ref_type = reference.get(corpus_name, None)
    if ref_type:
        ref_path = os.path.join(
            ref_dir,
            '.'.join([split, ref_type] + (['stem'] if stemmed else []) + ['json']))
    if not ref_type or not os.path.exists(ref_path):
        available_ref = set(f.split('.')[1] for f in os.listdir(ref_dir))
        logging.error('There is no "{}" reference for {} (Available : {})'.format(
            ref_type, corpus_name, ', '.join(available_ref)))
        return None
    return ref_path


################
# Caching temporary file for evaluation
################

stem = PorterStemmer().stem
# tokenize, stem, join, lower
tsjl = lambda s: ' '.join(map(stem, word_tokenize(s))).lower()

# Remove or add '.stem' to a file name, .stem is always the last information
#  before the extension
rem_stem = lambda path: path.replace('.stem', '')
add_stem = lambda path: '.'.join(path.split('.')[:-1] + ['stem'] +
                                 path.split('.')[-1:])


# Create a stemmed version of a reference file at `output_path`
def compute_stem(path, output_path):
    global current_writing_file
    # Should compute stemmed version and return the path of the file
    # If the output file already exist return it
    with open(path) as f:
        docs = json.load(f)
    docs = {k: [[tsjl(v) for v in kp] for kp in kws]
            for k, kws in docs.items()}
    current_writing_file = output_path
    with open(output_path, 'w') as g:
        json.dump(docs, g)
    current_writing_file = None
    return output_path


# Create a filtered version of a reference file at `output_path`
def filter_kws(path, data_path, output_path, key):
    # Use case is filtering absent/present keyphrases
    global current_writing_file
    with open(path) as f:
        docs = json.load(f)
    docs_filtered = {}
    
    with open(data_path) as f:
        for line in f:
            line = json.loads(line)
            doc_id = line['id']
            if doc_id not in docs:
                continue
            docs_filtered[doc_id] = [[v for v in kp if key(v, line['abstract'])]
                            for kp in docs[doc_id]]
            docs_filtered[doc_id] = [kp for kp in docs_filtered[doc_id] if kp]
    current_writing_file = output_path
    with open(output_path, 'w') as g:
        json.dump(docs_filtered, g)
    current_writing_file = None
    return output_path


# Create a stemmed version of a dataset file at `output_path`
def compute_tsjl(path, output_path):
    global current_writing_file
    logging.info('Computing stemmed documents for {}'.format(path))
    output_data = []
    with open(path) as f:
        nb_lines = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=nb_lines):
            line = json.loads(line)
            og_data = ' . '.join([line['title'], line['abstract']])
            og_data = tsjl(og_data)
            output_data.append(json.dumps(
                {'id': line['id'], 'abstract': og_data}) + '\n')
    current_writing_file = output_path
    with open(output_path, 'w') as g:
        for line in output_data:
            g.write(line)
    current_writing_file = None
    return output_path


# Return an existing path which is the stemmed version of a given reference
def stemmed(path):
    logging.debug('Searching for stemmed : ' + path)
    output_path = '/tmp/' + add_stem(path.replace(os.sep, '_'))
    # Already existing file
    if os.path.isfile(add_stem(path)):
        logging.debug('Found : ' + add_stem(path))
        return add_stem(path)
    # Temporary file
    elif os.path.isfile(output_path) and os.path.getmtime(output_path) > os.path.getmtime(path):
        logging.debug('Cached : ' + output_path)
        return output_path
    logging.debug('Not found : creating it')
    return compute_stem(path, output_path)


# Return an existing path which is the stemmed version of the given dataset file
def stem_data(path):
    logging.debug('Searching for docs stemmed : ' + path)
    output_path = '/tmp/' + path.replace(os.sep, '_') + '.tsjl'
    if os.path.isfile(output_path) and os.path.getmtime(output_path) > os.path.getmtime(path):
        logging.debug('Cached : ' + output_path)
        return output_path
    logging.debug('Not found : creating it')
    return compute_tsjl(path, output_path)


# Return an existing path which contain only absent keyphrases from the given
#  reference
def absent(path, data_path):
    logging.debug('Searching for absent : ' + path)
    output_path = '/tmp/' + path.replace(os.sep, '_') + '.abs'
    if os.path.isfile(output_path) and os.path.getmtime(output_path) > os.path.getmtime(path) and os.path.getmtime(output_path) > os.path.getmtime(data_path):
        logging.debug('Cached : ' + output_path)
        return output_path
    logging.debug('Not found : creating it')
    return filter_kws(path, data_path, output_path,
                      lambda kw, da: kw not in da)


# Return an existing path which contain only present keyphrases from the given
#  reference
def present(path, data_path):
    logging.debug('Searching for present : ' + path)
    output_path = '/tmp/' + path.replace(os.sep, '_') + '.prs'
    if os.path.isfile(output_path) and os.path.getmtime(output_path) > os.path.getmtime(path) and os.path.getmtime(output_path) > os.path.getmtime(data_path):
        logging.debug('Cached : ' + output_path)
        return output_path
    logging.debug('Not found : creating it')
    return filter_kws(path, data_path, output_path,
                      lambda kw, da: kw in da)
