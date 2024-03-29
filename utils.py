from itertools import chain
import re
import random
from collections import defaultdict, OrderedDict
import subprocess
import json
import os
import sys
import builtins
import ipdb
from copy import deepcopy
import traceback
import resource
import time
import pickle
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple
import csv
import glob

from tqdm import tqdm
import numpy as np
import torch
import h5py
import xlrd
import xlwt
import xlutils.copy
import psutil
import pytrec_eval
# from beir.retrieval.evaluation import EvaluateRetrieval

import metrics

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def get_git_revision_short_hash():
    """
    Returns the short Git revision hash of the current working directory.
    """
    gitpath = os.path.join(os.path.dirname(__file__), '.git')
    return subprocess.check_output(['git', '--git-dir', gitpath, 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')


def get_git_diff():
    """
    Returns the Git diff of the current working directory from HEAD revision.
    """
    gitpath = os.path.join(os.path.dirname(__file__), '.git')
    return subprocess.check_output(['git', '--git-dir', gitpath,'diff', 'HEAD', '--ignore-space-at-eol']).strip().decode('utf-8')


def write_conda_env(filepath):
    """Exports the packages installed in the current conda environment to a file."""
    # Also: os.system('conda list')
    return subprocess.check_output(['conda', 'env', 'export', '-f', filepath])


def rank_docs(docids, scores, shuffle=True):
    """Given a list of document IDs and a (potentially longer due to padding) 1D array of their scores, sort both scores
    and coresponding document IDs in the order of descending scores."""
    actual_scores = scores[:len(docids)]

    if shuffle:  # used to remove any possible bias (e.g. in case of score ties) in documents order
        inds = np.random.permutation(len(docids))
        actual_scores = actual_scores[inds]
        docids = [docids[i] for i in inds]

    inds = np.flip(np.argsort(actual_scores))
    actual_scores = actual_scores[inds]
    docids = [docids[i] for i in inds]
    return docids, actual_scores


def get_relevances(gt_relevant, candidates, max_docs=None, relevant_at_level=1):
    """Can handle multiple levels of relevance, including explicitly or implicitly 0 scores.
    Args:
        gt_relevant: for a given query, it's a dict mapping from ground-truth relevant candidate ID to level of relevance
        candidates: list of candidate pids
        max_docs: consider only the first this many documents
        relevant_at_level: any relevance lower than this threshold will be set to 0. This is for calculating binary metrics such as MRR.
    Returns: list of length min(max_docs, len(pred)) with non-zero relevance scores at the indices corresponding to passages in `gt_relevant`
        e.g. [0 2 1 0 0 1 0]
    """
    if max_docs is None:
        max_docs = len(candidates)
    return [gt_relevant[candid] if (candid in gt_relevant) and gt_relevant[candid] >= relevant_at_level else 0 for candid in candidates[:max_docs]]


# NOTE: except for MRR and Recall, metrics differ from official TREC. Use get_retrieval_metrics instead.
def calculate_metrics(relevances, num_relevant, k):
    eval_metrics = OrderedDict([('MRR', metrics.mean_reciprocal_rank(relevances)),
                                ('MRR@{}'.format(k), metrics.mean_reciprocal_rank(relevances, k)),
                                # ('MAP', metrics.mean_average_precision(relevances)),
                                # ('MAP@{}'.format(k), metrics.mean_average_precision(relevances, k)),
                                # ('MAP@R', metrics.mean_average_precision_at_r(relevances, num_relevant)),
                                ('Recall@{}'.format(k), metrics.recall_at_k(relevances, num_relevant, k)),
                                # ('nDCG', np.mean([metrics.ndcg_at_k(rel) for rel in relevances])),
                                # ('nDCG@{}'.format(k), np.mean([metrics.ndcg_at_k(rel, k) for rel in relevances]))
                                ])
    return eval_metrics


def write_predictions(filepath, predictions, format='tsv', score_type=builtins.float, is_qrels=False):
    """Writes score and rank predictions (or a qrels format with a column 'Q0' and no rank) to an output file of specified format.

    :param filepath: output file path
    :param predictions: dict of method's predictions per query, {str qID: {str pID: float score}}
    :param format: 'pickle', 'tsv' or 'hdf5'
    :param score_type: Only for format 'tsv': the type of the written scores (int or float).
    :param is_qrels: Only for format 'tsv': if True, will write 'tsv' file in qrels format with a column 'Q0' and no rank.
    """
    if format == 'tsv':
        if is_qrels:
            format_str = lambda qid, docid, rank, s: f"{qid}\tQ0\t{docid}\t{s}\n"
        else:
            format_str = lambda qid, docid, rank, s: f"{qid}\t{docid}\t{rank}\t{s}\n"
        with open(filepath, 'w') as out_file:
            for qid, doc2score in predictions.items():
                for i, (docid, score) in enumerate(doc2score.items()):
                    out_file.write(format_str(qid, docid, i+1, score_type(score)))
    elif format == 'pickle':
        with open(filepath, 'wb') as out_file:
            pickle.dump(predictions, out_file, protocol=pickle.HIGHEST_PROTOCOL)
    elif format == 'hdf5':
        with h5py.File(filepath, 'w') as out_file:
            write_dict_hdf5(out_file, predictions, dtype='float16')
    else:
        logger.debug("format '{}'; will not write output file")
    return


def write_dict_hdf5(group, d, dtype='float16'):
    """Write a dictionary to an HDF5 group; `group` starts as a file object,
    but subgroups will be created if necessary, to recursively write nested dictionaries.

    :param group: HDF5 group object (initially a file object)
    :param d: the dictionary to be written to disk
    """
    for key, value in d.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            write_dict_hdf5(subgroup, value, dtype)
        else:
            # group[key] = value  # this is used to write generic nested dictionaries, but doesn't specify dtype
            group.create_dataset(key, data=value, dtype=dtype)
    return


def read_dict_hdf5(group):
    """Read a dictionary from an HDF5 group; `group` starts as a file object,
    but existing subgroups will be extracted, to recursively read nested dictionaries.

    :param group: HDF5 group object (initially a file object)
    :return: the dictionary holding the data
    """
    d = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            d[key] = read_dict_hdf5(value)
        else:
            d[key] = value[()]
    return d


# DEPRECATED
def generate_rank(input_path, output_path):
    """
    Reads a file with scored candidate documents for each query, and writes a new file
    which sorts candidates per query by score and replaces score with rank
    """
    score_dict = defaultdict(list)
    for line in open(input_path):
        query_id, para_id, score = line.split("\t")
        score_dict[int(query_id)].append((float(score), int(para_id)))
    with open(output_path, "w") as outFile:
        for query_id, para_lst in score_dict.items():
            random.shuffle(para_lst)
            para_lst = sorted(para_lst, key=lambda x: x[0], reverse=True)
            for rank_idx, (score, para_id) in enumerate(para_lst):
                outFile.write("{}\t{}\t{}\n".format(query_id, para_id, rank_idx + 1))


# DEPRECATED
def eval_results(run_file_path,
                 eval_script="./ms_marco_eval.py",
                 qrels="./data/msmarco-passage/qrels.dev.small.tsv"):
    """Runs the MSMARCO evaluation script on a file with retrieved results and uses regex to find MRR in its output"""
    assert os.path.exists(eval_script) and os.path.exists(qrels)
    result = subprocess.check_output(['python', eval_script, qrels, run_file_path])
    match = re.search(r'MRR @10: ([\d.]+)', result.decode('utf-8'))
    mrr = float(match.group(1))
    return mrr


def set_seed(seed=None, has_gpu=False):
    """the seed state is shared across the entire program through importing `utils`, regardless of module
    (confirmed for Python random, but most likely true for the others too). Numpy is likely not thread safe."""
    random.seed(seed)
    np.random.seed(seed)
    if seed is None:
        seed = random.randint(0, 2**32-1)
    torch.manual_seed(seed)
    if has_gpu:
        torch.cuda.manual_seed_all(seed)


def save_HF_model_and_args(model, output_dir, save_name, args):
    """HuggingFace models only"""
    save_dir = os.path.join(output_dir, save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # first case is if model is wrapped in DataParallel
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(save_dir)  # only works with HF models
    torch.save(args, os.path.join(save_dir, 'training_args.bin'))


def save_model(path: str, global_step: int, model: object, optimizer: object = None, scheduler: object = None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    checkpoint = {'global_step': global_step,
                  'state_dict': state_dict}
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    torch.save(checkpoint, path)


def remove_oldest_checkpoint(dirpath, num_keep):
    """Removes oldest checkpoint, if existing checkpoints in `dirpath` are more than `num_keep`"""

    filelist = os.listdir(dirpath)
    suffices = map(lambda x: re.search(r"model_(\d+)\.", x), filelist)
    stepnums = sorted([int(matchobj.group(1)) for matchobj in suffices if matchobj])

    if len(stepnums) > num_keep:
        os.remove(os.path.join(dirpath, "model_{}.pth".format(stepnums[0])))
    return


def reverse_bisect_right(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted in descending order.

    The return value i is such that all e in a[:i] have e >= x, and all e in
    a[i:] have e < x.  So if x already appears in the list, a.insert(x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.

    Essentially, the function returns number of elements in a which are >= than x.
    >>> a = [8, 6, 5, 4, 2]
    >>> reverse_bisect_right(a, 5)
    3
    >>> a[:reverse_bisect_right(a, 5)]
    [8, 6, 5]
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if x > a[mid]:
            hi = mid
        else:
            lo = mid+1
    return lo


def load_model(model, model_path, device='cpu', resume=False, change_output=False):
    """
    :param model: an initialized model object, already on its intended device
    :param model_path: checkpoint file from which to load model
    :param device: Where to initially load the tensors. The device of 'model' will determine the final destination,
        but by explicitly setting this to the same device as where `model` resides, intermediate memory allocation may be avoided.
    :param resume: if True, will additionally load global_step, optimizer and scheduler states
    :param change_output: if True, the `output_layer` parameters will not be loaded onto the model (used for fine-tuning)

    :return: model: the model object, with weights loaded from a checkpoint
    :return: global_step: 0, unless `resume` is True, in which case the step number is loaded from a checkpoint
    :return: optimizer_state: None, unless `resume` is True, in which case a single state dict or a list of state dicts are loaded
    :return: scheduler_state: None, unless `resume` is True, in which case a single state dict or a list of state docts are loaded
    """
    global_step = 0
    checkpoint = torch.load(model_path, map_location=device)
    # checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
    if change_output:  # this is used when fine-tuning
        for key, val in list(checkpoint['state_dict'].items()):
            if key.startswith('output_layer'):
                checkpoint['state_dict'].pop(key)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info('Loaded model from {}. Global step: {}'.format(model_path, checkpoint['global_step']))

    optimizer_state, scheduler_state = None, None
    if resume:  # resume *training* from saved optimizer and scheduler
        try:
            global_step = checkpoint['global_step']
            optimizer_state = checkpoint['optimizer']
            if 'scheduler' in checkpoint:
                scheduler_state = checkpoint['scheduler']
        except KeyError:
            traceback.print_exc()
            logger.error("When `resume==True`, make sure that the states of an optimizer (and optionally scheduler) "
                         "exist in the checkpoint.")

    return model, global_step, optimizer_state, scheduler_state


def load_encoder(model, checkpoint_path, device='cpu'):
    """
    Loads query encoder weights from a CODER model checkpoint
    :param model: an initialized encoder model object (typically HuggingFace), already on its intended device
    :param checkpoint_path: MDSTransfomer checkpoint file from which to load encoder weights.
    :param device: Where to initially load the tensors. The device of 'model' will determine the final destination,
        but by explicitly setting this to the same device as where `model` resides, intermediate memory allocation may be avoided.

    :return: model: the model object, with weights loaded from a checkpoint
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
    for key, val in list(checkpoint['state_dict'].items()):
        if not(key.startswith('encoder')):  # discard parameter
            checkpoint['state_dict'].pop(key)
        else:  # keep parameter after trimming name
            new_key_name = key[len('encoder.'):]
            checkpoint['state_dict'][new_key_name] = checkpoint['state_dict'].pop(key)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    logger.info('Loaded model from {}. Global step: {}'.format(checkpoint_path, checkpoint['global_step']))

    return model


def move_to_device(obj, device):

    state = obj.state if hasattr(obj, 'state') else obj

    for param in state.values():
        # Not sure there are any global tensors in the state dict
        if torch.is_tensor(param):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            move_to_device(param, device)


def write_list(filepath, alist):

    with open(filepath, 'w') as f:
        for item in alist:
            f.write('{}\n'.format(item))


def stats_from_counts(counts, threshold=None, logger=None):
    """
    Given a list, iterable etc of numbers, will calculate statistics and plot histograms
    :param counts: iterable of numbers (e.g. number of tokens per sequence)
    :param threshold: shows how many numbers exceed threshold (e.g. sequences exceed the specified number in length)
    :param logger: a logger object to log output, otherwise "print" will be used
    """

    print_func = builtins.print if logger is None else logger.info

    mean = np.mean(counts)
    print_func("Mean: {}".format(mean))
    median = np.median(counts)
    print_func("Median: {}".format(median))

    if threshold is not None:
        num_above = np.sum(np.array(counts) > threshold)
        print_func("Above {}: {} ({:.3f}%)\n".format(threshold, num_above, 100*num_above/len(counts)))

    bin_edges = np.linspace(1, 2*threshold, num=30)
    freqs, bin_edges = np.histogram(counts, bins=bin_edges)
    bin_labels = ["[{:7.2f}, {:7.2f})".format(bin_edges[i], bin_edges[i + 1])
                  for i in range(len(bin_edges)-1)]
    logger.info('Histogram of frequencies:\n')
    ascii_bar_plot(bin_labels, freqs, width=50, logger=logger)

    return

# from: https://alexwlchan.net/2018/05/ascii-bar-charts/
def ascii_bar_plot(labels, values, width=30, logger=None):

    increment = max(values) / width
    longest_label_length = max(len(label) for label in labels)

    out_str = ''
    for label, value in zip(labels, values):

        # The ASCII block elements come in chunks of 8, so we work out how
        # many fractions of 8 we need.
        # https://en.wikipedia.org/wiki/Block_Elements
        bar_chunks, remainder = divmod(int(value * 8 / increment), 8)

        # First draw the full width chunks
        bar = '█' * bar_chunks

        # Then add the fractional part.  The Unicode code points for
        # block elements are (8/8), (7/8), (6/8), ... , so we need to
        # work backwards.
        if remainder > 0:
            bar += chr(ord('█') + (8 - remainder))

        # If the bar is empty, add a left one-eighth block
        bar = bar or '▏'

        out_str += f'{label.rjust(longest_label_length)} ▏ {value:#4d} {bar}\n'

    if logger is not None:
        logger.info('\n' + out_str)
    else:
        print(out_str)


def get_ranks_of_top(qrels, score_dict, hit=1000):
    """For each query, returns the rank (starting from 1) of the highest-ranking ground truth document.
    If more than one g.t. documents per query exist, they will be ignored.
    Thus, this estimates an upper bound of performance that reflects mainly MRR and not nDCG"""
    nonzero_score_indices = (np.nonzero(get_relevances(qrels[qid], list(score_dict[qid].keys()), max_docs=hit))[0] for qid in score_dict)
    ranks_top = np.array([1 + int(r[0]) if r.size else hit+1 for r in nonzero_score_indices])  # for each query, what was the rank of the highest-ranked rel. document
    return ranks_top


def get_ranks_of_all_relevant(qrels, score_dict, relevance_thr=1):
    """For each query, returns the ranks (starting from 1) of all corresponding ground-truth documents (or -1, if not found) as a list.
    Thus, the returned object is a list of lists."""

    def find(iterable, target):
        """Returns first index of target within iterable; if not found, returns -1"""
        for i, item in enumerate(iterable):
            if target == item:
                return i
        return -1

    return [[int(1 + find(score_dict[qid].keys(), docid)) for docid in qrels[qid] if qrels[qid][docid] >= relevance_thr] for qid in score_dict]


def plot_rank_histogram(qrels, pred_scores, base_pred_scores=None, include_ground_truth='all', bins=None, relevance_thr=1, save_as='ranks.pdf'):
    """
    Plots a histogram with the rank that ground-truth relevant documents ain `qrels` achieved
    based on the predicted model scores in `pred_scores`.

    :param qrels: dict {qID : {pID: score}, g.t. relevance
    :param pred_scores: dict {qID : {pID: score}, model predictions
    :param base_pred_scores: Optional: either dict {qID : {pID: score}, or tuple (freqs, bin_edges).
        If given, it will be superimposed on the plot for reference.
    :param include_ground_truth: if 'all', the histogram will include the ranks of all g.t. documents per query
        (naturally, only one of them can have rank 1). If 'top', the histogram will only include the ranks of the top-scored g.t. document.
    :param relevance_thr: the relevance score that a document in qrels must have in order to be g.t. considered relevant
    :param save_as: _description_, defaults to 'ranks.pdf'
    :return: (freqs, bin_edges) tuple of counts/frequencies and bin edges of pred_scores historgram
    """

    if include_ground_truth == 'all':
        title_str = 'Ranks of all ground-truth documents per query'
        pred_ranks = np.array(list(chain.from_iterable(get_ranks_of_all_relevant(qrels, pred_scores, relevance_thr=relevance_thr))), dtype=np.int16)
        if type(base_pred_scores) is dict:
            base_pred_ranks = np.array(list(chain.from_iterable(get_ranks_of_all_relevant(qrels, base_pred_scores, relevance_thr=relevance_thr))), dtype=np.int16)
    else:
        title_str= 'Ranks of highest-ranking ground-truth document per query'
        pred_ranks = np.array(list(chain.from_iterable(get_ranks_of_top(qrels, pred_scores))), dtype=np.int16)
        if type(base_pred_scores) is dict:
            base_pred_ranks = np.array(list(chain.from_iterable(get_ranks_of_top(qrels, base_pred_scores))), dtype=np.int16)

    bins= list(range(50)) + list(range(50, 1050, 50))
    # bin_labels = ["[{}, {})".format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)] + ["[{}, inf)".format(bins[-1])]
    plt.figure(figsize=(30, 10))
    if type(base_pred_scores) is dict:
        plt.hist(base_pred_ranks, bins=bins, alpha=0.2, color='red', label='Original')
    elif type(base_pred_scores) is tuple:
        freqs, bin_edges = base_pred_scores
        plt.hist(bin_edges[:-1], bins=bin_edges, weights=freqs, alpha=0.2, color='red', label='Original')
    freqs, bin_edges, _ = plt.hist(pred_ranks, bins=bins, alpha=0.2, color='blue', label='Reranked')
    plt.xticks(rotation='vertical')
    plt.xlabel('Rank')
    plt.ylabel('Counts')
    plt.legend()
    plt.title(title_str)
    plt.savefig(save_as)
    return freqs, bin_edges


def plot_rank_barplot(qrels, pred_scores, base_pred_scores=None, include_ground_truth='all', bins=None, relevance_thr=1, orient_bars='vertical', save_as='ranks.pdf'):
    """
    Plots a histogram with the rank that ground-truth relevant documents ain `qrels` achieved
    based on the predicted model scores in `pred_scores`. The difference with `plot_rank_histogram` is that
    the histogram is plotted as a horizontal barplot (i.e. bar the width is not proportional to the actual bin widths),
    with the y-axis being the rank and the x-axis being the frequency.

    :param qrels: dict {qID : {pID: score}, g.t. relevance
    :param pred_scores: dict {qID : {pID: score}, model predictions
    :param base_pred_scores: Optional: either dict {qID : {pID: score}, or tuple (freqs, bin_edges).
        If given, it will be superimposed on the plot for reference.
    :param include_ground_truth: if 'all', the histogram will include the ranks of all g.t. documents per query
        (naturally, only one of them can have rank 1). If 'top', the histogram will only include the ranks of the top-scored g.t. document.
    :param bins: iterable of length 69; the bins to use for the histogram. If None, the default bins will be used.
    :param relevance_thr: the relevance score that a document in qrels must have in order to be g.t. considered relevant
    :param orient_bars: orientation of bars; 'vertical' or 'horizontal', defaults to 'vertical'.
    :param save_as: filepath to save figure, defaults to 'ranks.pdf'
    :return: (freqs, bin_edges) tuple of counts/frequencies and bin edges of pred_scores historgram
    """

    if include_ground_truth == 'all':
        title_str = 'Ranks of all ground-truth documents per query'
        pred_ranks = np.array(list(chain.from_iterable(get_ranks_of_all_relevant(qrels, pred_scores, relevance_thr=relevance_thr))), dtype=np.int16)
        if type(base_pred_scores) is dict:
            base_pred_ranks = np.array(list(chain.from_iterable(get_ranks_of_all_relevant(qrels, base_pred_scores, relevance_thr=relevance_thr))), dtype=np.int16)
    else:
        title_str= 'Ranks of highest-ranking ground-truth document per query'
        pred_ranks = np.array(list(chain.from_iterable(get_ranks_of_top(qrels, pred_scores))), dtype=np.int16)
        if type(base_pred_scores) is dict:
            base_pred_ranks = np.array(list(chain.from_iterable(get_ranks_of_top(qrels, base_pred_scores))), dtype=np.int16)

    if bins is None:
        bins = list(range(1, 50)) + list(range(50, 1050, 50))
    bin_labels = ["[{}, {})".format(bins[i], bins[i + 1]) for i in range(len(bins) - 1)] #+ ["[{}, inf)".format(bins[-1])]

    if orient_bars == 'horizontal':
        fig, ax = plt.subplots(figsize=(10, 30))
    else:
        fig, ax = plt.subplots(figsize=(30, 10))

    if base_pred_scores is not None:
        if type(base_pred_scores) is dict:
            freqs, bin_edges = np.histogram(base_pred_ranks, bins=bins)
        elif type(base_pred_scores) is tuple:
            freqs, bin_edges = base_pred_scores
        orig_freqs = freqs
        # plt.hist(bin_edges[:-1], bins=bin_edges, weights=freqs, alpha=0.2, color='red', edgecolor='gray', label='Original')
        if orient_bars == 'horizontal':
            ax.barh(np.arange(len(freqs)), freqs, height=1, align='edge', alpha=0.2, color='red', edgecolor='gray', label='Original')
        else:
            ax.bar(np.arange(len(freqs)), freqs, width=1, align='edge', alpha=0.2, color='red', edgecolor='gray', label='Original')

    freqs, bin_edges = np.histogram(pred_ranks, bins=bins)
    # plt.hist(bin_edges[:-1], bins=bin_edges, weights=freqs, alpha=0.2, color='blue', edgecolor='gray', label='Reranked')

    # Prepare labels for the bars
    if base_pred_scores is None:
        label_vals = freqs
        # ax.bar_label(hbars, fmt='%d', padding=5)  # Requires matplotlib 3.4.0+
        axis_lim = max(freqs)
    else:
        label_vals = freqs - orig_freqs
        # ax.bar_label(hbars, fmt='%d', labels=labels, padding=10)  # Requires matplotlib 3.4.0+
        axis_lim = max(max(freqs), max(orig_freqs))

    axis_pos = np.arange(len(freqs))

    if orient_bars == 'horizontal':
        rects = ax.barh(axis_pos, freqs, height=1, align='edge', alpha=0.2, color='blue', edgecolor='gray', label='Reranked')
        ax.set_yticks(axis_pos)
        ax.set_yticklabels(bin_labels)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Counts')

        # Put labels on the right side of bars
        for rect, label_val in zip(rects, label_vals):
            width = rect.get_width()
            color = 'red' if label_val < 0 else 'blue'
            ax.text(width + 0.1*axis_lim, rect.get_y() + rect.get_height() / 2, str(label_val), ha="center", va="center", color=color)

        ax.set_xlim(right=1.2*axis_lim)  # adjust xlim to fit labels
    else:
        rects = ax.bar(axis_pos, freqs, width=1, align='edge', alpha=0.2, color='blue', edgecolor='gray', label='Reranked')
        ax.set_xticks(axis_pos)
        ax.set_xticklabels(bin_labels, rotation=90)
        ax.set_ylabel('Counts')

        # Put labels on top of bars
        for rect, label_val in zip(rects, label_vals):
            height = rect.get_height()
            color = 'red' if label_val < 0 else 'blue'
            ax.text(rect.get_x() + rect.get_width() / 2, height + 0.1*axis_lim, str(label_val), ha="center", va="center", color=color)

        ax.set_ylim(top=1.2*axis_lim)  # adjust ylim to fit labels

    ax.legend()
    ax.set_title(title_str)
    plt.savefig(save_as)
    return freqs, bin_edges


def write_columns_to_csv(filename, columns=None, rows=None, header=None, delimiter=','):
    """Writes the given columns (iterable of iterables of same length) XOR rows (iterable of iterables of arbitrary length)
    to a csv file by making use of the CSV writer module.
    If header is not None, it is written as the first line."""

    if columns is not None:
        if rows is not None:
            logger.warning('Both columns and rows were specified. Ignoring rows!')
        rows = zip(*columns)
    else:
        assert rows is not None, 'Either columns or rows must be specified!'

    with open(filename, 'w') as f:
        csvwriter = csv.writer(f, dialect='excel', delimiter=delimiter)
        if header is not None:
            csvwriter.writerow(header)
        csvwriter.writerows(rows)
    return


class Obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)


def dict2obj(d):
    """Convert dict `d` into an object ("struct")"""
    return json.loads(json.dumps(d), object_hook=Obj)


def load_config(args):
    """
    Returns a dictionary with the full experiment configuration settings.
    If a json file is specified with `--config`, its contents will overwrite the defaults or other arguments as
    extracted by argparse. If `--override '{'opt1':val1, 'opt2':val2}'` is used, then the dict-style formatted string
    will be used to override specified options.
    """

    config = deepcopy(args.__dict__)  # configuration dictionary

    if args.config_filepath is not None:
        logger.info("Reading configuration ...")
        try:  # dictionary containing the entire configuration settings in a hierarchical fashion
            with open(args.config_filepath) as cnfg:
                json_config = json.load(cnfg)
            config.update(json_config)
        except:
            logger.critical("Failed to load configuration file. Check JSON syntax and verify that files exist")
            traceback.print_exc()
            sys.exit(1)
        # Support command line override of config file options
        if args.override is not None:
            override_dict = eval(args.override)  # options to override
            config.update(override_dict)
        # Check for misspelled options in config file or command line override:
        unknown_config_keys = [k for k in config if not(k in args.__dict__)]
        if len(unknown_config_keys):
            raise ValueError("The following options/keys are not supported: {}".format(unknown_config_keys))

    return config


def create_dirs(dirs):
    """
    Input:
        dirs: a list of directories to create, in case these directories are not found
    Returns:
        exit_code: 0 if success, -1 if failure
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def export_performance_metrics(filepath, metrics_table, header, book=None, sheet_name="metrics"):
    """Exports performance metrics on the validation set for all epochs to an excel file"""

    if book is None:
        book = xlwt.Workbook()  # new excel work book

    book = write_table_to_sheet([header] + metrics_table, book, sheet_name=sheet_name)

    book.save(filepath)
    logger.info("Exported per epoch performance metrics in '{}'".format(os.path.realpath(filepath)))

    return book


def write_row(sheet, row_ind, data_list):
    """Write a list to row_ind row of an excel sheet"""

    row = sheet.row(row_ind)
    for col_ind, col_value in enumerate(data_list):
        row.write(col_ind, col_value)
    return


def write_table_to_sheet(table, work_book, sheet_name=None):
    """Writes a table implemented as a list of lists to an excel sheet in the given work book object"""

    sheet = work_book.add_sheet(sheet_name)

    for row_ind, row_list in enumerate(table):
        write_row(sheet, row_ind, row_list)

    return work_book


def export_record(filepath, values):
    """Adds a list of values as a bottom row of a table in a given excel file"""

    read_book = xlrd.open_workbook(filepath, formatting_info=True)
    read_sheet = read_book.sheet_by_index(0)
    last_row = read_sheet.nrows

    work_book = xlutils.copy.copy(read_book)
    sheet = work_book.get_sheet(0)
    write_row(sheet, last_row, values)
    work_book.save(filepath)


def register_record(filepath, timestamp, experiment_name, best_metrics, final_metrics=None, parameters=None, comment=''):
    """
    Adds the best and final metrics of a given experiment as a record in an excel sheet with other experiment records.
    Creates excel sheet if it doesn't exist.
    Args:
        filepath: path of excel file keeping records
        timestamp: string
        experiment_name: string
        best_metrics: dict of metrics at best epoch {metric_name: metric_value}. Includes "epoch" as first key
        final_metrics: dict of metrics at final epoch {metric_name: metric_value}. Includes "epoch" as first key
        parameters: dict of hyperparameters {param_name: param_value}
        comment: optional description
    """
    metrics_names, metrics_values = zip(*best_metrics.items())
    row_values = [timestamp, experiment_name, comment]
    if parameters is not None:
        param_names, param_values = zip(*parameters.items())
        row_values += list(param_values)
    row_values += list(metrics_values)
    if final_metrics is not None:
        final_metrics_names, final_metrics_values = zip(*final_metrics.items())
        row_values += list(final_metrics_values)
    if not os.path.exists(filepath):  # Create a records file for the first time
        logger.warning("Records file '{}' does not exist! Creating new file ...".format(filepath))
        directory = os.path.dirname(filepath)
        if len(directory) and not os.path.exists(directory):
            os.makedirs(directory)
        header = ["Timestamp", "Name", "Comment"]
        if parameters is not None:
            header += param_names
        header += ["Best " + m for m in metrics_names]
        if final_metrics is not None:
            header += ["Final " + m for m in final_metrics_names]

        book = xlwt.Workbook()  # excel work book
        book = write_table_to_sheet([header, row_values], book, sheet_name="records")
        book.save(filepath)
    else:
        try:
            export_record(filepath, row_values)
        except Exception as x:
            alt_path = os.path.join(os.path.dirname(filepath), "record_" + experiment_name)
            logger.error("Failed saving in: '{}'! Will save here instead: {}".format(filepath, alt_path))
            export_record(alt_path, row_values)
            filepath = alt_path

    logger.info("Exported performance record to '{}'".format(os.path.realpath(filepath)))


#from https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py
class EvaluateRetrieval:

    @staticmethod
    def evaluate(qrels: Dict[str, Dict[str, int]],
                 results: Dict[str, Dict[str, float]],
                 k_values: List[int],
                #  relevance_level=1,
                 ignore_identical_ids: bool=True,
                 verbose=True) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        :param qrels: Dict[query_id, Dict[pasasge_id, relevance_score]] ground truth
        :param results: Dict[query_id, Dict[pasasge_id, relevance_score]] predictions
        :param k_values: iterable of integer cut-off thresholds
        # :param relevance_level: relevance score in qrels which a doc should at least have in order to be considered relevant
        :param ignore_identical_ids: ignore identical query and document ids (default)
        :param verbose: it True, will use logger config to print metrics
        :return: Dict[str, float] value for each metric (determined by `k_values`)
        """

        if ignore_identical_ids:
            # logging.info('For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this.')
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}

        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string}) #relevance_level=relevance_level) aparrently gives error
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]

        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)

        if verbose:
            for eval in [ndcg, _map, recall, precision]:
                logging.info("\n")
                for k in eval.keys():
                    logging.info("{}: {:.4f}".format(k, eval[k]))

        return ndcg, _map, recall, precision

    @staticmethod
    def evaluate_custom(qrels: Dict[str, Dict[str, int]],
                        results: Dict[str, Dict[str, float]],
                        k_values: List[int],
                        metric: str,
                        relevance_level=1,
                        verbose=True) -> Tuple[Dict[str, float]]:

        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            return metrics.mrr(qrels, results, k_values, relevance_level=relevance_level, verbose=verbose)
        else:
            raise NotImplementedError(f"Metric '{metric}' not implemented")
        # elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
        #     return recall_cap(qrels, results, k_values)

        # elif metric.lower() in ["hole", "hole@k"]:
        #     return hole(qrels, results, k_values)

        # elif metric.lower() in ["acc", "top_k_acc", "accuracy", "accuracy@k", "top_k_accuracy"]:
        #     return top_k_accuracy(qrels, results, k_values)


def get_retrieval_metrics(results, qrels, cutoff_values=(1, 3, 5, 10, 100, 1000), relevance_level=1, verbose=True):
    """Compute retrieval performance metrics with parity to the official TREC implementation.

    :param results: dict of method's predictions per query, {str qID: {str pID: float score}}
    :param qrels: dict of ground truth relevance judgements per query, {str qID: {str pID: int relevance}}
    :param cutoff_values: interable of @k cut-offs for metrics calculation, defaults to (1, 3, 5, 10, 100, 1000)
    :param relevance_level: relevance score in qrels which a doc should at least have in order to be considered relevant. Only for MRR
    :return: dict of aggregate metrics, {"metric@k": avg. metric value}
    """

    intersect_qids = results.keys() & qrels.keys()
    if len(intersect_qids) < len(qrels) or len(intersect_qids) < len(results):
        logger.warning(f"qrels file contains rel. labels for {len(qrels)} queries, while {len(results)} "
                       f"queries were evaluated! Performance metrics will only consider the intersection ({len(intersect_qids)} queries)!")
        logger.warning("{} queries in reference but not in predictions".format(len(qrels.keys() - results.keys())))
        logger.warning("{} queries in predictions but not in reference".format(len(results.keys() - qrels.keys())))
        qrels = {qid: qrels[qid] for qid in intersect_qids}
        results = {qid: results[qid] for qid in intersect_qids}

    logger.info("Computing metrics ...")
    start_time = time.perf_counter()
    # Evaluate using pytrec_eval for comparison with BEIR benchmarks
    # Returns dictionaries with metrics for each cut-off value, e.g. ndcg["NDCG@{k}".format(cutoff_values[0])] == 0.3
    metric_dicts = EvaluateRetrieval.evaluate(qrels, results, cutoff_values, ignore_identical_ids=False, verbose=verbose)  # tuple of metrics dicts (ndct, ...) #TODO: RESTORE
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, cutoff_values, 'MRR', relevance_level=relevance_level, verbose=verbose)
    metrics_time = time.perf_counter() - start_time
    logger.info("Time to calculate performance metrics: {:.3f} s".format(metrics_time))

    # Merge all dicts into a single OrderedDict and sort by k for guaranteed consistency
    perf_metrics = OrderedDict()  # to also work with Python 3.6
    for met_dict in (mrr, ) + metric_dicts: #TODO: RESTORE
        perf_metrics.update(sorted(met_dict.items(), key=lambda x: 0 if '@' not in x[0] else int(x[0].split('@')[1])))

    return perf_metrics


class Printer(object):
    """Class for printing output by refreshing the same line in the console, e.g. for indicating progress of a process"""

    def __init__(self, console=True):

        if console:
            self.print = self.dyn_print
        else:
            self.print = builtins.print

    @staticmethod
    def dyn_print(data):
        """Print things to stdout on one line, refreshing it dynamically"""
        sys.stdout.write("\r\x1b[K" + data.__str__())
        sys.stdout.flush()


def readable_time(time_difference):
    """Convert a float measuring time difference in seconds into a tuple of (hours, minutes, seconds)"""

    hours = time_difference // 3600
    minutes = (time_difference // 60) % 60
    seconds = time_difference % 60

    return hours, minutes, seconds


def check_model(model, verbose=False, zero_thresh=1e-8, inf_thresh=1e6, stop_on_error=False):
    status_ok = True
    for name, param in model.named_parameters():
        param_ok = check_tensor(param, verbose=verbose, zero_thresh=zero_thresh, inf_thresh=inf_thresh)
        if not param_ok:
            status_ok = False
            print("Parameter '{}' PROBLEM".format(name))
        grad_ok = True
        if param.grad is not None:
            grad_ok = check_tensor(param.grad, verbose=verbose, zero_thresh=zero_thresh, inf_thresh=inf_thresh)
        if not grad_ok:
            status_ok = False
            print("Gradient of parameter '{}' PROBLEM".format(name))
        if stop_on_error and not (param_ok and grad_ok):
            ipdb.set_trace()

    if status_ok:
        print("Model Check: OK")
    else:
        print("Model Check: PROBLEM")


def check_tensor(X, verbose=True, zero_thresh=1e-8, inf_thresh=1e6):
    is_nan = torch.isnan(X)
    if is_nan.any():
        print("{}/{} nan".format(torch.sum(is_nan), X.numel()))
        return False

    num_small = torch.sum(torch.abs(X) < zero_thresh)
    num_large = torch.sum(torch.abs(X) > inf_thresh)

    if verbose:
        print("Shape: {}, {} elements".format(X.shape, X.numel()))
        print("No 'nan' values")
        print("Min: {}".format(torch.min(X)))
        print("Median: {}".format(torch.median(X)))
        print("Max: {}".format(torch.max(X)))

        print("Histogram of values:")
        values = X.view(-1).detach().numpy()
        hist, binedges = np.histogram(values, bins=20)
        for b in range(len(binedges) - 1):
            print("[{}, {}): {}".format(binedges[b], binedges[b + 1], hist[b]))

        print("{}/{} abs. values < {}".format(num_small, X.numel(), zero_thresh))
        print("{}/{} abs. values > {}".format(num_large, X.numel(), inf_thresh))

    if num_large:
        print("{}/{} abs. values > {}".format(num_large, X.numel(), inf_thresh))
        return False

    return True


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def recursively_hook(model, hook_fn):
    for name, module in model.named_children():  # model._modules.items():
        if len(list(module.children())) > 0:  # if not leaf node
            for submodule in module.children():
                recursively_hook(submodule, hook_fn)
        else:
            module.register_forward_hook(hook_fn)


def get_current_memory_usage():
    """RSS Memory usage in MB"""
    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]
    return int(memusage.strip()) / 1024


def get_current_memory_usage2():
    """RSS Memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info()[0] / (1024 ** 2)


def get_max_memory_usage():
    """max RSS Memory usage in MB"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class Timer(object):

    def __init__(self):
        self.total_time = 0.0
        self.count = 0

    def update(self, measured_time):
        self.total_time += measured_time
        self.count += 1
        return

    def get_average(self):
        if self.count:
            return self.total_time / self.count
        else:
            return None


def load_qrels(filepath, relevance_level=1, score_mapping=None, rel_type=builtins.int, id_type=builtins.str):
    """Load ground truth relevant passages (or scores per passage) from file. Can handle several levels of relevance.
    Assumes that if a passage is not listed for a query, it is non-relevant.
    :param filepath: path to file of ground truth relevant passages in the following format ('Q0' can be missing):
        "qID1 \t Q0 \t pID1 \t 2\n
         qID1 \t Q0 \t pID2 \t 0\n
         qID1 \t Q0 \t pID3 \t 1\n..."
    :param relevance_level: only include candidates which have at least the specified relevance score
        (after potential mapping)
    :param score_mapping: dictionary mapping relevance scores in qrels file to a different value (e.g. 2 -> 0.3)
    :param rel_type: type of relevance scores (int or float)
    :param id_type: type of query and passage IDs (int or str)
    :return:
        qid2relevance (dict): dictionary mapping from query_id to relevant passages (dict {passageid : relevance})
    """

    qid2relevance = defaultdict(dict)
    with open(filepath, 'r') as f:
        for line in tqdm(f, desc="Line: "):
            try:
                fields = line.strip().split()
                if len(fields) == 4:
                    qid, _, pid, relevance = fields
                else:
                    qid, pid, relevance = fields
                relevance = rel_type(relevance)
                if (score_mapping is not None) and (relevance in score_mapping):
                    relevance = score_mapping[relevance]  # map score to new value
                if relevance >= relevance_level:  # include only if score >= specified relevance level
                    qid2relevance[id_type(qid)][id_type(pid)] = relevance
            except Exception as x:
                print(x)
                raise IOError("'{}' is not valid format".format(line))
    return qid2relevance


def load_predictions(filepath, seperator=None):
    """
    Load retrieved document/passage IDs from a file with their respective scores, if they exist.
    If scores are not explicitly given (as a 4th field), fictitious values are used.
    Assumes that retrieved documents per query are given in the order of rank (most relevant first) in the first 2
    columns as "qID1 \t pID1 \t [rank] \t [score]\n qID1 \t pID2\n ..."  but not necessarily contiguously (sorted by qID).
    :param filepath: path to file of candidate (retrieved) documents/passages per query
    :return:
        qid_to_candidate_passages: dict: {qID : {pID: score}}
    """

    qid_to_candidate_passages = defaultdict(dict)  # dict: {qID : {pID: score}}
    score = 100  # initialize fake score value, required for TREC evaluation tools if scores not explicitly given

    with open(filepath, 'r') as f:
        for line in f:
            fields = line.strip().split(seperator)
            qid = fields[0]
            pid = fields[1]
            try:
                score = float(fields[3])
            except IndexError:
                score = 0.999 * score  # ensures score will be decreasing
            qid_to_candidate_passages[qid][pid] = score

    return qid_to_candidate_passages


def merge_qrels_dictionaries(dicts, overwrite_queries=False):
    """Merge multiple qrel dictionaries, in a way that new documents can be added to existing shared queries,
    and if for the same query a passage is present in multiple dictionaries, the value from the last will be used.

    :param dicts: iterable of qrels dictionaries to merge. Each is a dict of the form {qid: {pid: rel}}
    :param overwrite_queries: if True, queries that are present in multiple dictionaries will be overwritten by the last;
        otherwise, they will be merged such that new passages can be added to existing queries (and shared passages will be overwritten).
    """
    merged = defaultdict(dict)

    if overwrite_queries:
        for d in dicts:
            merged.update(d)
    else:
        for d in dicts:
            for qid, pid2rel in d.items():
                merged[qid].update(pid2rel)
    return merged


def load_qrels_from_pickles(path_pattern, overwrite_queries=False):
    """Load qrels dictionaries from multiple pickle files, and merge them into a single dictionary.
    :param path_pattern: path pattern to pickle files, e.g. "/path/to/qrels/*.pickle"
    :param overwrite_queries: if True, queries that are present in multiple dictionaries will be overwritten by the last;
        otherwise, they will be merged such that new passages can be added to existing queries (and shared passages will be overwritten).
    """
    qrels = defaultdict(dict)
    paths = glob.glob(path_pattern)
    if len(paths) == 0:
        raise IOError("No pickle files found for pattern: '{}'".format(path_pattern))
    for path in glob.glob(path_pattern):
        with open(path, 'rb') as f:
            if overwrite_queries:
                qrels.update(pickle.load(f))
            else:
                for qid, pid2rel in pickle.load(f).items():
                    qrels[qid].update(pid2rel)
    return qrels


def load_qrels_from_hdf5(path_pattern, overwrite_queries=False):
    """Load qrels dictionaries from multiple HDF5 files, and merge them into a single dictionary.
    :param path_pattern: path pattern to HDF5 files, e.g. "/path/to/qrels/*.hdf5"
    :param overwrite_queries: if True, queries that are present in multiple dictionaries will be overwritten by the last;
        otherwise, they will be merged such that new passages can be added to existing queries (and shared passages will be overwritten).
    """
    qrels = defaultdict(dict)
    paths = glob.glob(path_pattern)
    if len(paths) == 0:
        raise IOError("No HDF5 files found for pattern: '{}'".format(path_pattern))
    for path in glob.glob(path_pattern):
        with h5py.File(path, "r") as f:
            if overwrite_queries:
                qrels.update(read_dict_hdf5(f))
            else:
                for qid, pid2rel in read_dict_hdf5(f).items():
                    qrels[qid].update(pid2rel)
    return qrels
