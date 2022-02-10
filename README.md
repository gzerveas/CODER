# CODER: An efficient framework for improving retrieval through COntextualized Document Embedding Reranking

This codebase implements CODER, a framework that improves the performance of existing retrieval models by training them through efficient, contextual reranking.

As a starting point, this code was initially forked from: https://github.com/jingtaozhan/RepBERT-Index.

In the following, we will show how to train and evaluate a CODER model using [TAS-B](https://dl.acm.org/doi/10.1145/3404835.3462891)  (Sebastian Hofstaetter et al., 2021) as a base method.

## Setup

_Instructions refer to Unix-based systems (e.g. Linux, MacOS)._

In the following, let's assume that you have cloned the code inside a directory called `coder`.

You need to create/choose a directory as root, e.g. `~/Experiments`. Inside this root directory, each experiment will create a time-stamped output directory, which contains
model checkpoints, performance metrics per epoch, the experiment configuration, log files, predictions per query, etc.

[We recommend creating and activating a `conda` or other Python virtual environment (e.g. `virtualenv`) to 
install packages and avoid conficting package requirements; otherwise, to run `pip`, the flag `--user` or `sudo` privileges will be necessary.]

`pip install -r coder/requirements.txt`


## Downloading data

Download `collectionandqueries.tar.gz` from [MSMARCO-Passage-Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking). It contains passages, queries, and qrels.

```bash
mkdir data
cd data
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar xvfz collectionandqueries.tar.gz
```

`collectionandqueries.tar.gz` should have MD5 checksum of `31644046b18952c1386cd4564ba2ae69`.

## Pre-processing data

#### - Filter queries:
The MS MARCO `queries.train.tsv` and `queries.dev.tsv` files also contain queries for which relevance labels are not available.
To eliminate unnecessary waiting times (e.g. for retrieval of candidates) or ID look-up errors when training, you can filter
them in this way:

```bash
# Read IDs from first column of first file; for the second, only print a line if its first column (i.e. ID) is contained in the first file
awk '(NR==FNR){a[$1];next} ($1 in a){print $0}' qrels.train.tsv queries.train.tsv > queries.in_qrels.train.tsv
```

#### - Tokenize collection & queries:
For efficiency, collection documents (i.e. passages) and queries are first kept pre-tokenized and numerized in JSON files.

```bash
python ~/coder/convert_text_to_tokenized.py --output_dir . --collection collection.tsv --queries . --tokenizer_from "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco" 
```

#### - Create memmap arrays for collection (mapping doc_ID -> doc_token_IDs) 

```bash
python ~/coder/create_memmaps.py --tokenized_collection collection.tokenized.json --output_collection_dir collection_memmap --max_doc_length 256 
```

#### - Compute document representations:
(~2h for 8.8M doc of MS MARCO)

```bash
python ~/coder/precompute.py --model_type huggingface --encoder_from "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco" --collection_memmap_dir  collection_memmap/ --output_dir . 
```

### Obtain candidate documents to be used as negatives for each query

The following 2 steps are required in order to obtain the top TAS-B candidates per query. Some publicly available methods (e.g. RepBERT) make their predictions per query directly available for download, in which case the following 2 steps are not necessary.

#### - Compute query representations:
(~186 q/sec - 5.4 ms/query)

```bash
python ~/coder/precompute.py --model_type huggingface --encoder_from "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco" --output_dir . --tokenized_queries queries.$SET.json --per_gpu_batch_size 100
```

The above should be run for `SET={in_qrels.train, in_qrels.dev, dev.small}` (or `in_qrels.train` and `in_qrels.dev`).

#### - Retrieve 1000 candidates per query through dense retrieval:
(4.15 queries/sec for searching through 8.8M docs)

Running dense retrieval for all queries in the training set is the lengthiest step of the pipeline.
We can parallelize it by splitting the queries set in a number of smaller parts, e.g. 5:
```bash
split -l $((`wc -l < queries.in_qrels.train.tsv`/5)) queries.in_qrels.train.tsv query_ids.in_qrels.train --additional-suffix=".tsv" -da 1
```

Now, we can start one retrieval job for each piece (each should take a few hours):

```bash
python ~/coder/retrieve.py --doc_embedding_dir doc_embeddings_memmap/ --query_embedding_dir queries.in_qrels.train_embeddings_memmap --query_ids query_ids.in_qrels.train$1.txt --output_path tasb_top1000.train.$1.tsv --hit 1000 --per_gpu_doc_num 3000000 
```

and eventually concatenate the results: `cat tasb_top1000.train.*.tsv > tasb_top1000.train.tsv`.

We also need to run retrieval for the validation (and testing) sets:

```bash
python ~/coder/retrieve.py --doc_embedding_dir doc_embeddings_memmap/ --query_embedding_dir queries.in_qrels.dev_memmap --output_path tasb_top1000.dev.tsv --hit 1000 --per_gpu_doc_num 3000000 
```

and

```bash
python ~/coder/retrieve.py --doc_embedding_dir doc_embeddings_memmap/ --query_embedding_dir queries.dev.small_memmap --output_path tasb_top1000.dev.small.tsv --hit 1000 --per_gpu_doc_num 3000000 
```

[The option `--query_ids queries.in_qrels.dev.txt` can be used if we haven't already filtered IDs when precomputing query representations.]

#### - Create memmap arrays for query-candidates (mapping query_ID -> candidate_doc_IDs):

```bash
python ~/coder/create_memmaps.py --candidates tasb.top1000.train.tsv --output_candidates_dir tasb.top1000.train_memmap 
```

The above step should also be executed for `dev` and `dev.small`.

## Training and Evaluating CODER

Because of numerous options, CODER is best trained and evaluated using a configuration file, as follows:
```bash
python ~/coder/main.py --config ~/coder/configurations/coder_tasb_train_config.json
```

We include the configuration files for training and evaluating CODER(TAS-B) in `~coder\configurations`.

However, it is also possible to use commandline options, either instead of a configuration file, 
or to override some options with the option `--override`, e.g.: 
```bash
python ~/coder/main.py --config ~/coder/coder_tasb_train_config.json --override '{"learning_rate": 1e-6}'
```

A detailed documentation of all options can be found as follows (or inside `~/coder/options.py`):

### Show and explain all options for training, evaluation, inspection
```bash
python ~/coder/main.py --help
```

### Computing evaluation metrics

We can use CODER(TAS-B) for reranking TAS-B on `dev.small` by running:
```bash
python ~/coder/main.py --config ~/coder/configurations/coder_tasb_eval_config.json
```

This will output CODER's rankings for each query inside `~/Experiments/DEMO_NAME/predictions`, and also display several evaluation metrics.
However, to obtain authoritative evaluation metrics (esp. for nDCG), please use [trec_eval](https://trec.nist.gov/trec_eval/).

First, we need to convert the rankings file format to `.trec`:
```bash
awk '{print $1 " Q0 " $2 " " $3 " " $4 " coder"}' ~/Experiments/DEMO_NAME/predictions/*.tsv > coder_reranked_tasb_top1000.dev.small.trec
```

Finally, run the `trec_eval` executable:
```bash
trec_eval -m all_trec  qrels.dev.small.tsv reranked_tasb_top1000.dev.small.trec
```

## Using CODER for dense retrieval

After we have trained CODER, we use the stored checkpoint inside `~/Experiments/DEMO_NAME/checkpoints` 
to compute the query representations for the set of interest, e.g. `dev.small`:

```bash
mkdir CODER_rep
python ~/coder/precompute.py --model_type mdst_transformer --encoder_from "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco" --load_checkpoint ~/Experiments/DEMO_NAME/checkpoints/model_best.pth --output_dir CODER_rep --tokenized_queries queries.dev.small.json --per_gpu_batch_size 256
```

Finally, we perform dense retrieval:

```bash
python ~/coder/retrieve.py --doc_embedding_dir doc_embeddings_memmap/ --query_embedding_dir Coder_rep/queries.dev.small_memmap --output_path coder_top1000.dev.tsv --hit 1000 --per_gpu_doc_num 3000000
```
