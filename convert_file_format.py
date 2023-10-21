import logging
from operator import itemgetter
logging.basicConfig(format='%(asctime)s | %(name)-8s - %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()
import os
import argparse
import pickle
import builtins

import torch
import h5py
from tqdm import tqdm

import utils


parser = argparse.ArgumentParser("Convert one file format to another. The defaults correspond to converting a tsv scores file to a pickle file used for training.")
# Run from config fule
parser.add_argument('--from', dest='orig_filepath', required=True,
                    help='Source file path')
parser.add_argument('--to', dest='dest_filepath',
                    help='Destination file path, whose extention will determine the destination format. '
                    'If this argument is not specified, it will be the same as the source file path, but with the opposite extension (.tsv or .pickle).')
parser.add_argument('--score_type', choices=['int', 'float'], default='float',
                    help='Type of destination (prediction or label) scores')
parser.add_argument('--id_type', choices=['int', 'str'], default='int',
                    help='Type of destination query or document IDs')
parser.add_argument('--is_qrels', action='store_true',
                    help="If the destination 'tsv' file is meant to be in a qrels format (including a 'Q0' column and no rank column).")
parser.add_argument('--num_top', type=int, default=None,
                    help="If set, will write only the `num_top` documents for each query (in descending order of score).")
parser.add_argument('--norm_relevances', type=str, choices=['None', 'max', 'maxmin', 'std'], default='None',
                    help="How to normalize the relevance scores. Supposed to increase dynamic range and/or limit absolute range.")
args = parser.parse_args()


def normalize_scores(scores, normalize):
    """
    Normalization is supposed to help dealing with the variety of score ranges resulting from different hyperparameter combinations.
    Ultimately, a KL divergence between the model-predicted score distribution and the target relevance distribution will be computed,
    and the target rel. dist. will be obtained by applying a softmax (with temperature) on relevances computed here.

    :param scores: (n,) tensor of relevance scores
    :param normalize: str, how to normalize: 'None', 'max', 'maxmin', 'std'
    :return: (n,) tensor of normalized relevance scores
    """
    if normalize == 'max':  # in [0, 1]. May result in fairly flat "distribution" close to 1.
        scores = scores / torch.max(scores)
    elif normalize == 'maxmin':  # increases dynamic range for "uniform" initial scores, while still in [0, 1]
        min_relev = torch.min(scores)
        scores = (scores - min_relev) / (torch.max(scores) - min_relev)
    elif normalize == 'std':  # even wider dynamic range for "uniform" initial scores, but in [0, f], with f > 1
        relev_std = torch.std(scores)
        scores = (scores - torch.min(scores)) / relev_std
    elif normalize != 'None':  # 'None' results in arbitrary range of scores, with potentially flat "distribution".
        raise ValueError(f"Unknown relevance score normalization option '{normalize}'")
    return scores


class Converter(object):

    def __init__(self, orig_filepath, dest_filepath=None, score_type='float', id_type='int', is_qrels=False, num_top=None, norm_relevances='None'):
        """
        :param orig_filepath: input/origin file path
        :param dest_filepath: output/destination file path. If not specified, it will be the same as the source file path, but with the opposite extension (.tsv or .pickle).
        :param score_type: the type of the written scores (int or float).
        :param id_type: type of query and passage IDs (int or str)
        :param is_qrels: Only for format 'tsv': if True, will write 'tsv' file in qrels format with a column 'Q0' and no rank.
        :param num_top: If set, will write only the `num_top` documents for each query (in descending order of score).
        :param norm_relevances: if not 'None', will normalize the relevance scores. Supposed to increase dynamic range and/or limit absolute range.
        """
        self.orig_filepath = orig_filepath
        self.dest_filepath = dest_filepath
        self.score_type = builtins.int if score_type == 'int' else builtins.float
        self.id_type = builtins.int if id_type == 'int' else builtins.str  # for pickle format
        self.is_qrels = is_qrels
        self.num_top = num_top
        self.norm_relevances = norm_relevances

        # Determine the source and destination formats
        orig_main_filename, self.orig_ext = os.path.splitext(self.orig_filepath)
        if self.dest_filepath is not None:  # when specified, the destination format is determined by the extension
            self.dest_format = os.path.splitext(self.dest_filepath)[1][1:]  # [1:] to remove the dot
        else:
            if self.orig_ext == '.pickle' or self.orig_ext == '.hdf5':  # default conversion when origin not .tsv
                self.dest_format = 'tsv'
            else:  # default conversion for origin '.tsv'
                self.dest_format = 'pickle' #'hdf5'
            self.dest_filepath = orig_main_filename + '.' + self.dest_format

        # Set the types of the IDs according to the destination format
        if self.dest_format == 'pickle':
            self.is_qrels = False
        elif self.dest_format == 'hdf5':
            self.id_type = builtins.str  # only string keys are supported in HDF5 format
            self.is_qrels = False

        return

    def convert(self):
        """Reads a file with scored candidate documents for each query into a dictionary {qid: {passageid: relevance}}
        and writes it to a new binary or tsv file, according to the specified format."""

        logger.info("Reading scores from {} ...".format(self.orig_filepath))
        if self.orig_ext == '.tsv':
            scores = utils.load_qrels(self.orig_filepath, relevance_level=0, rel_type=self.score_type, id_type=self.id_type)  # dict{qID: dict{pID: relevance}}
        elif self.orig_ext == '.pickle':
            with open(self.orig_filepath, 'rb') as f:
                scores = pickle.load(f)  # dict{qID: dict{pID: relevance}}
        else:
            with h5py.File(self.orig_filepath, 'r') as f:
                # scores = utils.read_dict_hdf5(f)  # dict{qID: dict{pID: relevance}}
                scores = {qid: {pid: self.score_type(relevance) for pid, relevance in f[qid].items()} for qid in f}

        if self.num_top is not None or self.norm_relevances != 'None':
            if self.num_top is not None:
                logger.info("Will keep only the top {} passages for each query".format(self.num_top))
            if self.norm_relevances != 'None':
                logger.info(f"Will normalize relevance scores with '{self.norm_relevances}'")
            for qid in tqdm(scores, desc="Processing query: ", total=len(scores)):
                end = self.num_top if self.num_top is not None else len(scores[qid])
                if self.norm_relevances != 'None':
                    docids, s = zip(*sorted(scores[qid].items(), key=itemgetter(1), reverse=True)[:end])
                    s = normalize_scores(torch.tensor(s, dtype=torch.float16), self.norm_relevances)
                    scores[qid] = dict((docids[i], s[i]) for i in range(len(docids)))
                    # NOTE: to remove boost factor from first passage score. RESTORE BY REMOVING BELOW!
                    # first_docid = next(iter(scores[qid].items()))[0]
                    # scores[qid][first_docid] /= 2
                else:
                    scores[qid] = dict(sorted(scores[qid].items(), key=itemgetter(1), reverse=True)[:end])

        logger.info("Writing labels/scores to {} ...".format(self.dest_filepath))
        utils.write_predictions(self.dest_filepath, scores, format=self.dest_format, score_type=self.score_type, is_qrels=self.is_qrels)

        return


def convert_tsv_to_pickle(input_path, output_path, score_type, id_type, num_top=None):
    """
    Reads a file with scored candidate documents for each query into a
    dictionary {qid: {passageid: relevance}} and writes it to a new binary file.

    :param input_path: input tsv file path
    :param output_path: output pickle file path
    :param score_type: Only for format 'tsv': the type of the written scores (int or float).
    :param id_type: type of query and passage IDs (int or str)
    """

    score_type = builtins.int if score_type == 'int' else builtins.float
    id_type = builtins.int if id_type == 'int' else builtins.str

    logger.info("Reading scores from {} ...".format(input_path))
    scores = utils.load_qrels(input_path, relevance_level=0, rel_type=score_type, id_type=id_type)

    if num_top is not None:
        logger.info("Keeping only the top {} passages for each query ...".format(num_top))
        for qid in scores:
            scores[qid] = dict(sorted(scores[qid].items(), key=itemgetter(1), reverse=True)[:num_top])
            # NOTE: to remove boost factor from first passage score. RESTORE BY REMOVING BELOW!
            # first_docid = next(iter(scores[qid].items()))[0]
            # scores[qid][first_docid] /= 2

    logger.info("Writing labels/scores to {} ...".format(output_path))
    utils.write_predictions(output_path, scores, format="pickle")
    return


def convert_pickle_to_tsv(input_path, output_path, score_type, is_qrels, num_top=None):
    """
    Reads a binary file containing a dictionary {qid: {passageid: relevance}} with scored candidate documents
     and writes it to a new tsv file.

    :param input_path: input pickle file path
    :param output_path: output tsv file path
    :param score_type: Only for format 'tsv': the type of the written scores (int or float).
    :param is_qrels: Only for format 'tsv': if True, will write 'tsv' file in qrels format with a column 'Q0' and no rank.
    """

    score_type = builtins.int if score_type == 'int' else builtins.float
    # id_type = builtins.int if id_type == 'int' else builtins.str

    logger.info("Reading scores from {} ...".format(input_path))
    with open(input_path, 'rb') as f:
        scores = pickle.load(f)  # dict{qID: dict{pID: relevance}}

    if num_top is not None:
        logger.info("Keeping only the top {} passages for each query ...".format(num_top))
        for qid in scores:
            scores[qid] = dict(sorted(scores[qid].items(), key=itemgetter(1), reverse=True)[:num_top])

    logger.info("Writing labels/scores to {} ...".format(output_path))
    utils.write_predictions(output_path, scores, format="tsv", score_type=score_type, is_qrels=is_qrels)
    return


if __name__ == "__main__":

    converter = Converter(args.orig_filepath, args.dest_filepath,
                          args.score_type, args.id_type, args.is_qrels, args.num_top, args.norm_relevances)
    converter.convert()

    logger.info("Done!")
