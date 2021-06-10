
# augment

Classical text data augmentation.

Provides automatic debiasing and class imbalance correction by sampling from various sources and performing various word replacement strategies. A modular tool currently implemented for English and German, and easily expandable.

## features

- **Data Generation via Replacement**
  - Strategies: Perturbation, Semantic Relations
  - Sources: WordNet, GermaNet, Word Lists

- **Data Generation via Sampling**
  - Sources: Twitter, Twint, MioWordCorpus
  - Debiasing Sampling: Find samples that balance features (words/concepts) that occur frequently in the positive class but are not related to the classification task. For example:
    1. Search for neutral terms that are overrepresented in the positive class
    2. Search for examples of those terms that are neutral/negative class

- **Backlogs: Rule Out Duplicate Samples/Augmentations**
  - Main backlog: Records job configs, doesn’t re-run the same config
  - Samples backlog: Records sample IDs, doesn’t include a sample if it has already been processed before

## disclaimer

This project is from 2020 and there are a few things to be noted.

- Twitter API and the Twint package are not operational anymore
- GermaNet: needs a license, is served locally using MongoDB, and accessed using a client predating the official one.
- Our custom ```dataload``` library is used to load data. This will need to be replaced with your own loader.

## install

- Dependencies: ```pip install twint tweepy langid sqlite3 requests tqdm editdistance nltk git+https://github.com/hexis-systems/dataload.git git+https://github.com/hexis-systems/pygermanet.git```
- WordNet: ```python -m nltk.downloader wordnet```
- GermaNet: ```python -m pygermanet.mongo_import ~/GN_V80/GN_V80_XML/```

## run

- ```./augment.py```

- Input: _config.json_
    - multiple jobs can be defined
    - the backlog is checked to see if a specific config has already been executed

- Configuration file format
  - *dataset_name* (string)
  - *input_path* (list): path(s) to input file(s)
  - *data_loader* (string): name of loader function in ```dataload```
  - *augmentations* (list): ```perturbations```, ```synonyms```, ```debiasing```
  - *sample_source* (string): ```twitter```, ```twint```, ```miowordcorpus```
  - *output_path* (string): path to output directory
  - *language_code* (string): ```de```, ```en```

- Output
    - Data: txt file(s) in TSV format
    - Metadata: config + samples backlog in JSON files

## good to know

- Regenerate existing augmentations
  1. manual deletion of: relevant entries in _backlog.json_, corresponding txt files, and entries in _debiasing-samples-backlog.json_
  2. restart augment.py with the same config

- A new language requires
    - word lists: keywords, stop words
    - wordnet: multiwordnet, babelnet, or similar
    - (optional) a classification model for your task

- Sampling with twint produces some duplicates
    - some output files are therefore empty
    - possible solutions
        - sample more frequently at different times
        - search larger volumes and match with the samples backlog as early as possible
