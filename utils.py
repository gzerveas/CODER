import re
import random
from collections import defaultdict
import subprocess
import json
import os
import sys
import builtins
import ipdb
from copy import deepcopy
import traceback
import resource

import numpy as np
import torch
import xlrd
import xlwt
from xlutils.copy import copy
import psutil

import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_rank(input_path, output_path):
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
    suffices = map(lambda x: re.search(r"model_(\d*)\.", x), filelist)
    stepnums = sorted([int(matchobj.group(1)) for matchobj in suffices if matchobj])

    if len(stepnums) >= num_keep:
        os.remove("model_{}.cpt".format(stepnums[0]))

    return


def load_model(model, model_path, optimizer=None, scheduler=None, resume=False, change_output=False):
    global_step = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
    state_dict = deepcopy(checkpoint['state_dict'])
    if change_output:  # this is used when fine-tuning
        for key, val in checkpoint['state_dict'].items():
            if key.startswith('output_layer'):
                state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    logger.info('Loaded model from {}. Global step: {}'.format(model_path, checkpoint['global_step']))

    # resume *training* from saved optimizer and scheduler 
    if resume:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            global_step = checkpoint['global_step']

            lrs = []  # list of learning rates, one per parameter group
            for param_group in optimizer.param_groups:
                lrs.append(param_group['lr'])
            logger.info('Resumed optimizer with learning rate(s): {}'.format(lrs))
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
                logger.info('Resumed scheduler with learning rate(s): {}'.format(scheduler.get_last_lr()))
        except Exception:
            traceback.print_exc()
            logger.error("""When `resume==True`, make sure that an initialized optimizer (and optionally scheduler) 
            has been passed as an argument to `load_model`, and that the respective state(s) exist in the checkpoint.""")

    return model, global_step, optimizer, scheduler


def move_to_device(obj, device):

    state = obj.state if hasattr(obj, 'state') else obj

    for param in state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            move_to_device(param, device)


class Obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)


def dict2obj(d):
    """Convert dict `d` into an object ("struct")"""
    return json.loads(json.dumps(d), object_hook=Obj)


def load_config(config_filepath):
    """
    Using a json file with the master configuration (config file for each part of the pipeline),
    return a dictionary containing the entire configuration settings in a hierarchical fashion.
    """

    with open(config_filepath) as cnfg:
        config = json.load(cnfg)

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

    work_book = copy(read_book)
    sheet = work_book.get_sheet(0)
    write_row(sheet, last_row, values)
    work_book.save(filepath)


def register_record(filepath, timestamp, experiment_name, best_metrics, final_metrics=None, comment=''):
    """
    Adds the best and final metrics of a given experiment as a record in an excel sheet with other experiment records.
    Creates excel sheet if it doesn't exist.
    Args:
        filepath: path of excel file keeping records
        timestamp: string
        experiment_name: string
        best_metrics: dict of metrics at best epoch {metric_name: metric_value}. Includes "epoch" as first key
        final_metrics: dict of metrics at final epoch {metric_name: metric_value}. Includes "epoch" as first key
        comment: optional description
    """
    metrics_names, metrics_values = zip(*best_metrics.items())
    row_values = [timestamp, experiment_name, comment] + list(metrics_values)
    if final_metrics is not None:
        final_metrics_names, final_metrics_values = zip(*final_metrics.items())
        row_values += list(final_metrics_values)

    if not os.path.exists(filepath):  # Create a records file for the first time
        logger.warning("Records file '{}' does not exist! Creating new file ...".format(filepath))
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        header = ["Timestamp", "Name", "Comment"] + ["Best " + m for m in metrics_names]
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
