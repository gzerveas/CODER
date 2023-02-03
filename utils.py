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

import numpy as np
import torch
import xlrd
import xlwt
import xlutils.copy
import psutil
from beir.retrieval.evaluation import EvaluateRetrieval

import metrics

import logging
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def rank_docs(docids, scores, shuffle=True):
    """Given a list of document IDs and a (potentially longer due to padding) 1D array of their scores, sort both scores
    and coresponding document IDs in the order of descending scores."""
    actual_scores = scores[:len(docids)]

    if shuffle:  # used to remove any possible bias in documents order
        inds = np.random.permutation(len(docids))
        actual_scores = actual_scores[inds]
        docids = [docids[i] for i in inds]

    inds = np.flip(np.argsort(actual_scores))
    actual_scores = actual_scores[inds]
    docids = [docids[i] for i in inds]
    return docids, actual_scores


def get_relevances(gt_relevant, candidates, max_docs=None):
    """Can handle multiple levels of relevance, including explicitly or implicitly 0 scores.
    Args:
        gt_relevant: for a given query, it's a dict mapping from ground-truth relevant passage ID to level of relevance
        candidates: list of candidate pids
        max_docs: consider only the first this many documents
    Returns: list of length min(max_docs, len(pred)) with non-zero relevance scores at the indices corresponding to passages in `gt_relevant`
        e.g. [0 2 1 0 0 1 0]
    """
    if max_docs is None:
        max_docs = len(candidates)
    return [gt_relevant[pid] if pid in gt_relevant else 0 for pid in candidates[:max_docs]]


def calculate_metrics(relevances, num_relevant, k):
    eval_metrics = OrderedDict([('MRR@{}'.format(k), metrics.mean_reciprocal_rank(relevances, k)),
                                ('MAP@{}'.format(k), metrics.mean_average_precision(relevances, k)),
                                ('Recall@{}'.format(k), metrics.recall_at_k(relevances, num_relevant, k)),
                                ('nDCG@{}'.format(k), np.mean([metrics.ndcg_at_k(rel, k) for rel in relevances])),

                                ('MRR', metrics.mean_reciprocal_rank(relevances)),
                                ('MAP', metrics.mean_average_precision(relevances)),
                                ('nDCG', np.mean([metrics.ndcg_at_k(rel) for rel in relevances]))])
    return eval_metrics


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


def eval_results(run_file_path,
                 eval_script="./ms_marco_eval.py",
                 qrels="./data/msmarco-passage/qrels.dev.small.tsv"):
    """Runs the MSMARCO evaluation script on a file with retrieved results and uses regex to find MRR in its output"""
    assert os.path.exists(eval_script) and os.path.exists(qrels)
    result = subprocess.check_output(['python', eval_script, qrels, run_file_path])
    match = re.search(r'MRR @10: ([\d.]+)', result.decode('utf-8'))
    mrr = float(match.group(1))
    return mrr


def set_seed(args):
    """the seed state is shared across the entire program, regardless of module
    (confirmed for Python random, but most likely true for the others too). Numpy is likely not thread safe."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


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
    Loads query encoder weights from a MDSTransformer model checkpoint
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
    logger.info("Exported per epoch performance metrics in '{}'".format(filepath))

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

    logger.info("Exported performance record to '{}'".format(filepath))


def get_retrieval_metrics(results, qrels, cutoff_values=(1, 3, 5, 10, 100, 1000)):
    """Compute retrieval performance metrics with parity to the official TREC implementation. 

    :param results: dict of method's predictions per query, {str qID: {str pID: float score}}
    :param qrels: dict of ground truth relevance judgements per query, {str qID: {str pID: int relevance}}
    :param cutoff_values: interable of @k cut-offs for metrics calculation, defaults to (1, 3, 5, 10, 100, 1000)
    :return: dict of aggregate metrics, {"metric@k": avg. metric value}
    """
    
    if len(results) < len(qrels):
        logger.warning(f"qrels file contains rel. labels for {len(qrels)} queries, while only {len(results)} "
                    "queries were evaluated. Performance metrics will only consider the intersection.")
        qrels = {qid: qrels[qid] for qid in results.keys()}  # trim qrels dict
    
    logger.info("Computing metrics ...")
    start_time = time.perf_counter()
    # Evaluate using BEIR (which relies on pytrec_eval) for comparison with BEIR benchmarks
    # Returns dictionaries with metrics for each cut-off value, e.g. ndcg["NDCG@{k}".format(cutoff_values[0])] == 0.3
    cutoff_values = [1, 3, 5, 10, 100, 1000]
    metric_dicts = EvaluateRetrieval.evaluate(qrels, results, cutoff_values)  # tuple of metrics dicts (ndct, ...)
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, cutoff_values, 'MRR')
    metrics_time = time.perf_counter() - start_time
    logger.info("Time to calculate performance metrics: {:.3f} s".format(metrics_time))
    
    # Merge all dicts into a single OrderedDict and sort by k for guaranteed consistency
    perf_metrics = OrderedDict()
    for met_dict in (mrr, ) + metric_dicts:
        perf_metrics.update(sorted(met_dict.items(), key=lambda x: 0 if not '@' in x[0] else int(x[0].split('@')[1])))

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
    return int(memusage.strip())/1024


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


def load_qrels(filepath, relevance_level=1, score_mapping=None):
    """Load ground truth relevant passages from file. Can handle several levels of relevance.
    Assumes that if a passage is not listed for a query, it is non-relevant.
    :param filepath: path to file of ground truth relevant passages in the following format:
        "qID1 \t Q0 \t pID1 \t 2\n
         qID1 \t Q0 \t pID2 \t 0\n
         qID1 \t Q0 \t pID3 \t 1\n..."
    :param relevance_level: only include candidates which have at least the specified relevance score
        (after potential mapping)
    :param score_mapping: dictionary mapping relevance scores in qrels file to a different value (e.g. 2 -> 0.3)
    :return:
        qid2relevance (dict): dictionary mapping from query_id to relevant passages (dict {passageid : relevance})
    """
    qid2relevance = defaultdict(dict)
    with open(filepath, 'r') as f:
        for line in f:
            try:
                qid, _, pid, relevance = line.strip().split()
                relevance = int(relevance)
                if (score_mapping is not None) and (relevance in score_mapping):
                    relevance = score_mapping[relevance]  # map score to new value
                if relevance >= relevance_level:  # include only if score >= specified relevance level
                    qid2relevance[qid][pid] = relevance
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