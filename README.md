# Large-Scale Evaluation of Keyphrase Extraction Models

This repository holds the code necessary to reproduce results from the paper "Large-Scale Evaluation of Keyphrase Extraction Models" accepted at JCDL2020.

This table shows the f-score @ top 10 (F@10).

| model            | PubMed | ACM | SemEval-2010 | Inspec | WWW | KP20k | DUC-2001 | 500N-KPCrowd | KPTimes | NYTime |
|:-----------------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| FirstPhrases     |   15.4 |   13.6 |   13.8 |   29.3 |   10.2 |   13.5 |   24.6 |   17.1 |   11.4 |    9.2 |
| TextRank         |    1.8 |    2.5 |    3.5 |   35.8 |    8.4 |   10.2 |   21.5 |    7.1 |    2.8 |    2.7 |
| TfIdf            |   16.7 |   12.1 |   17.7 |   36.5 |    9.3 |   *11.5* |   23.3 |   16.9 |   12.4 |    9.6 |
| PositionRank     |    4.9 |    5.7 |    6.8 |   34.2 |   11.6 |   14.1 |   28.6 |   13.4 |   10.4 |    8.5 |
| MultipartiteRank |   15.8 |   11.6 |   14.3 |   30.5 |   10.8 |   13.6 |   25.6 |   18.2 |   14.0 |   11.2 |
| EmbedRank        |    3.7 |    2.1 |    2.5 |   35.6 |   10.7 |   12.4 |   29.5 |   12.4 |    4.7 |    *3.1* |
| Kea              |   18.6 |   14.2 |   19.5 |   34.5 |   11.0 |   14.0 |   26.5 |   17.3 |   13.8 |   11.0 |
| CopyRNN          |   24.2 |   24.4 |   20.3 |   28.2 |   22.2 |   *25.5* |   12.7 |   15.5 |   14.9 |   11.0 |
| CopyCorrRNN      |   20.8 |   21.1 |   19.4 |   27.9 |   19.9 |   *22.0* |   17.0 |   11.5 |   11.9 |    9.7 |
| CopyRNN_News     |   11.6 |    5.1 |    7.0 |    9.2 |    6.3 |    6.6 |   10.5 |    8.4 |   31.9 |   39.3 |
| CopyCorrRNN_News | n/a    | n/a    | n/a    | n/a    | n/a    | n/a    |   10.5 |    7.8 |   19.8 |   *20.5* |


## Requirements
- [pke](https://github.com/boudinfl/pke)
	- Install with `python3 -m pip install git+https://github.com/boudinfl/pke`
	- To execute EmbedRank you will need [sent2vec_wiki_bigrams](https://drive.google.com/open?id=0B6VhzidiLvjSaER5YkJUdWdPWU0) (16GB !) downloadable from [epfml/sent2vec](https://github.com/epfml/sent2vec#downloading-sent2vec-pre-trained-models)
	- To execute CopyRNN and CopyCorrRNN you will need [CopyRNN pretrained]() and [CorrRNN pretrained]()
- [ake-datasets](https://github.com/boudinfl/ake-datasets)
	- Clone with `git clone https://github.com/boudinfl/ake-datasets`
	- Define environment variable `export PATH_AKE_DATASET=PATH/TO/ake-datasets`
	- You will need [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/#download)
	- Define environment variable `export PATH_CORENLP=PATH/TO/stanford-corenlp-full-...`
	- Preprocess datasets by running `_preprocess.sh` for each dataset (this can take a while for large dataset)
	- KP20k and KPTimes are downloaded automatically when running `_preprocess.sh` but you can start downloading now with these links:
		- [KP20k](https://drive.google.com/open?id=1Z7fgWkmGaVElhH9tuf08p1vVoZBkAKbk) (214MB)
		- [KPTimes Test](https://drive.google.com/open?id=1LSREXfJxAK2jbzzvXYUvXufPdvL_Aq1J) (30MB)
		- [KPTimes Valid](https://drive.google.com/open?id=1XgVZbIw0Cbs2ZczBj2-tUKg3Z2whi5Zm) (19MB)
		- [KPTimes Train](https://drive.google.com/open?id=12chZA87VUviFyOh1qWs8DI33hbjKsKiv) (474MB)

## Running models

To run keyphrase extraction models on each dataset:

```bash
bash _benchmarks.sh
```

The output will be stored in `output/DATASET/DATASET.MODEL(.stem)?.json`.
You can change which models are executed by editing corresponding `params/DATASET.json` file.

## Evaluating

Evaluate one specific output:

`python3 evaluation/eval.py -i output/DATASET/DATASET.MODEL.stem.json -r $PATH_AKE_DATASETS/datasets/DATASET/references/REF_TYPE.test.stem.json`

Evaluate all outputs and create a `.csv` holding all scores:

`python3 evaluation/evaluate_all.py -v output scores.csv`

Using `python3 evaluation/make_tables.py scores.csv` will output a table (like the one in this README).



## Citing this paper
Large-Scale Evaluation of Keyphrase Extraction Models. [[arXiv](https://arxiv.org/pdf/2003.04628.pdf), [code](https://github.com/ygorg/JCDL_2020_KPE_Eval)]
Ygor Gallina, Florian Boudin, Béatrice Daille.
Joint Conference on Digital Libraries (JCDL), 2020.
