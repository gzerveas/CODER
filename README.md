# Improving Dense Retrieval by Training through Contextual Reranking

This codebase implements methods that improve dense retrieval by taking into account **ranking context**: that is, a large set of documents, positives and negatives, that are all closely related to a specific query and are jointly scored for relevance.

The core method is [CODER](https://arxiv.org/abs/2112.08766), a framework that **improves the performance of existing retrieval models by fine-tuning them through efficient, contextual reranking**. In particular, only the query encoder is fine-tuned through a list-wise loss, while the document embeddings are pre-computed for the entire collection and remain unchanged.

The same code additionally implements the bias mitigation method described in the paper: [Mitigating Bias in Search Results Through Contextual Document Reranking and Neutrality Regularization](https://dl.acm.org/doi/10.1145/3477495.3531891). Essentially, list-wise training with CODER is augmented by an additional loss term that penalizes the model when assigning a high relevance score to documents which are biased with respect to a given attribute (e.g. gender). Thus, **the model learns to promote documents that are both relevant and more neutral with respect to the same attribute**.

Finally, this codebase implements the method described in the paper [Enhancing the Ranking Context of Dense Retrieval Methods through Reciprocal Nearest Neighbors](https://arxiv.org/abs/2305.15720). This method **improves the performance of dense retrieval models by enhancing the ranking context**. This is accomplished through evidence-based label smoothing, i.e. by propagating relevance from the ground-truth documents to unlabeled documents that are intimately connected to them.

If you find this code helpful, please consider citing the respective papers:

```
@inproceedings{zerveas-etal-2022-coder,
    title = "{CODER}: An efficient framework for improving retrieval through {CO}ntextual Document Embedding Reranking",
    author = "Zerveas, George  and
      Rekabsaz, Navid  and
      Cohen, Daniel  and
      Eickhoff, Carsten",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.727",
    doi = "10.18653/v1/2022.emnlp-main.727",
    pages = "10626--10644",
}
```

```
@inproceedings{10.1145/3477495.3531891,
author = {Zerveas, George and Rekabsaz, Navid and Cohen, Daniel and Eickhoff, Carsten},
title = {Mitigating Bias in Search Results Through Contextual Document Reranking and Neutrality Regularization},
year = {2022},
isbn = {9781450387323},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3477495.3531891},
doi = {10.1145/3477495.3531891},
location = {Madrid, Spain},
series = {SIGIR '22}
}
```

```
@misc{zerveas2023enhancing,
      title={Enhancing the Ranking Context of Dense Retrieval Methods through Reciprocal Nearest Neighbors}, 
      author={George Zerveas and Navid Rekabsaz and Carsten Eickhoff},
      year={2023},
      eprint={2305.15720},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```


[As a starting point, this code was initially forked from: https://github.com/jingtaozhan/RepBERT-Index.]

## Contents

1. [Setup and data preparation](#setup)
2. [Training and evaluating CODER](#training-and-evaluating-coder)
3. [Using trained CODER models shared online](#using-already-trained-coder-models)
4. [Reranking with reciprocal nearest neighbors](#reranking-with-reciprocal-nearest-neighbors)
5. [Training with evidence-based label smoothing](#training-with-evidence-based-label-smoothing)
6. [Training for bias mitigation](#training-for-bias-mitigation)

## Setup

_Instructions refer to Unix-based systems (e.g. Linux, MacOS)._

In the following, let's assume that you have cloned the code inside a directory called `coder`.

You need to create/choose a directory as root, e.g. `~/Experiments`. Inside this root directory, each experiment will create a time-stamped output directory, which contains
model checkpoints, performance metrics per epoch, the experiment configuration, log files, predictions per query, etc.

[We recommend creating and activating a `conda` or other Python virtual environment (e.g. `virtualenv`) to 
install packages and avoid conflicting package requirements; otherwise, to run `pip`, the flag `--user` or `sudo` privileges will be necessary.]

This code has been tested with Python 3.6, so you can create an Anaconda environment with: `conda create -n CODER python=3.6`

`pip install -r coder/requirements.txt`


## Downloading data

### MS MARCO

Download `collectionandqueries.tar.gz` from [MSMARCO-Passage-Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking). It contains passages, queries, and qrels.

```bash
mkdir data
cd data
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar xvfz collectionandqueries.tar.gz
```

`collectionandqueries.tar.gz` should have MD5 checksum of `31644046b18952c1386cd4564ba2ae69`.

### TripClick

Step-by-step instructions on how to download TripClick, as well as details about the dataset contents, are found here: https://tripdatabase.github.io/tripclick/.

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
(~2h for 8.8M doc of MS MARCO on a single GPU, faster if more GPUs are available)

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

We note that the present repository does not support triplet training from scratch; instead, it fine-tunes (and thus relies on the existence of) a trained retrieval model (e.g. available as a pytorch checkpoint or on HuggingFace), which is called the _base method_.

In the following, we will show how to train and evaluate a CODER model using [TAS-B](https://dl.acm.org/doi/10.1145/3404835.3462891)  (Sebastian Hofstaetter et al., 2021) as a base method.

**If you want to use an already trained CODER model, please go to section:** [Using already trained CODER models](#using-already-trained-coder-models).


Because of numerous options, CODER is best trained and evaluated using a configuration file, as follows:
```bash
python ~/coder/main.py --config ~/coder/configs/<coder_tasb_train_config>.json
```

We include the configuration files for training and evaluating CODER(TAS-B) in `~coder\configs`.

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
python ~/coder/main.py --config ~/coder/configs/TEST_<coder_tasb_eval_config>.json
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

## Using already trained CODER models

Two CODER retriever models, already trained on MS MARCO using TAS-B and CoCondenser as the base model (i.e. fine-tuning starting point), are available on the HuggingFace Hub:
[CODER-TAS-B](https://huggingface.co/gzerveas/CODER-TAS-B) and [CODER-CoCondenser](https://huggingface.co/gzerveas/CODER-CoCondenser).

You can use them in your code as follows:

```python
from transformers import AutoModel, AutoTokenizer

query_encoder = AutoModel.from_pretrained('gzerveas/CODER-TAS-B')
tokenizer = AutoTokenizer.from_pretrained('gzerveas/CODER-TAS-B')
```

Thus, instead of training CODER, in case you want to use these trained models to precompute query representations in order to perform dense retrieval on an entire set, the command given in the immediately preceding section will now become:

```bash
python ~/coder/precompute.py --model_type mdst_transformer --encoder_from "gzerveas/CODER-TAS-B" --output_dir CODER_rep --tokenized_queries queries.dev.small.json --per_gpu_batch_size 256
```

Please note that CODER only trains the query encoder and therefore the document encoders (as well as tokenizers for queries and documents) will be the same as the ones of the base model, i.e. [TAS-B](https://huggingface.co/sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco) or [CoCondenser](https://huggingface.co/Luyu/co-condenser-marco-retriever), and can be used with the options `--query_encoder_from <HF_string_ID>` or `--encoder_from <HF_string_ID>` in the commands above, or directly downloaded and used through HuggingFace API, e.g. `AutoModel.from_pretrained('sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco')`.


## Reranking with Reciprocal Nearest Neighbors

Here we show how we can use Reciprocal Nearest Neighbors to **rerank the top candidates retrieved by an arbitrary dense retrieval model in post-processing**. For this, we only need the script `reciprocal_compute.py`. Here we will use the same resources as [above](#pre-processing-data), but generally one only needs pre-computed query and document embeddings by any dense retrieval model.


After preparing the document and query embeddings and retrieving the initial set of 1000 candidates
per query (see [above](#pre-processing-data)), we can rerank them by a similarity metric based on Reciprocal Nearest Neibghbors using the following command:

```bash
python ~/coder/reciprocal_compute.py --task rerank --query_embedding_dir $QUERY_EMB"queries.dev_embeddings_memmap/" --query_ids ~/data/MS_MARCO/queries.dev.small.tsv --doc_embedding_dir $DOC_EMB --output_dir $PRED_PATH"dev.small" --exp_name $RNN_ALIAS"_hit"60 --hit 60 --candidates_path  $CANDIDATES_PATH"top1000.dev.small.tsv" --qrels_path ~/data/MS_MARCO/qrels.dev.small.tsv --device cpu --k 21 --trust_factor 0.12756543122778 --k_exp 5 --sim_mixing_coef 0.468765448351993 --normalize None --weight_func linear --records_file $RECORDS_PATH"_rerank_MSMARCO.dev.small_records.xls" --write_to_file pickle
```


## Training with evidence-based label smoothing

Here we show how we can use evidence-based label smoothing to **train an arbitrary dense retrieval model**. To generate the new labels, we only need to run the script `reciprocal_compute.py`. As resources, we need the original relevance labels from the IR dataset, and pre-computed query and document embeddings produced by the dense retrieval model of our choice. Here we will use the same resources as [above](#pre-processing-data).

First, we need to obtain the smooth labels, which are computed based on a linear combination of geometric similarity and Jaccard similarity using reciprocal nearest neighbors:

```bash
python ~/coder/reciprocal_compute.py --task smooth_labels --query_embedding_dir "$QUERY_EMB"queries.train_embeddings_memmap/ --doc_embedding_dir $DOC_EMB --output_dir ~/RecipNN/smooth_labels/label_files/ --exp_name rNN_EBsmooth_train --candidates_path $CANDIDATES_PATH  --qrels_path ~/data/MS_MARCO/qrels.train.tsv  --hit 60 --k 21 --trust_factor 0.12756543122778 --k_exp 5 --sim_mixing_coef 0.468765448351993 --normalize None --weight_func linear --rel_aggregation mean --redistribute radically --return_top 63 --no_prefix --write_to_file tsv
```

Above, we have chosen to output the labels into a `.tsv` file, which is suitable for large datasets. Instread, we can choose to write the labels into a `.pickle` file, which is faster, but may run out of memory. 


We then simply have to specify the output `.pickle`  or `.tsv` file as an input for option `target_scores_path` in the CODER training configuration file. We also have to specify score transformation hyperparameters, such as the boost factor. All the relevant
hyperparameters, including those for CODER training, are included in the JSON configuration file inside : `configs`.

Finally, we can train simply by running:

```bash
python ~/coder/main.py --config ~/coder/configs/{your_config}.config
```

## Training for bias mitigation

To train CODER for bias mitigation, the process is exactly the same as described [above](#training-and-evaluating-coder). Additinonally, we will need the resources in the directory `fair_retrieval`. In particular, we will first need to compute neutrality scores for all collection documents by using the script `fair_retrieval/resources/calc_documents_neutrality.py`, which will write neutrality scores for each document to a file in tsv format (docid [tab] score).

Then, when starting CODER training (as usual, through `main.py`), we need to specify the following additional options in the training configuration file (or command line):

```
collection_neutrality_path: path to the file containing neutrality values of documents in tsv format (docid [tab] score)
background_set_runfile_path: path to the TREC run file containing the documents of the background set. Provided: fair_retrieval/resources/msmarco_fair.background_run.txt (for MSMARCO), or fair_retrieval/resources/trec2019_fair.bm25.txt (for TREC DL 2019).
bias_regul_coeff: coefficient of the bias regularization term added to the training loss
bias_regul_cutoff: number of top retrieved documents for which the bias term is calculated
```
