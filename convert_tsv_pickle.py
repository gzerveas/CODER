import logging
from operator import itemgetter
logging.basicConfig(format='%(asctime)s | %(name)-8s - %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()
import os
import argparse
import pickle

from tqdm import tqdm

import utils



parser = argparse.ArgumentParser("Convert one file format to another. The defaults correspond to converting a scores tsv file to a pickle file used for training.")
# Run from config fule
parser.add_argument('--from', dest='orig_filepath', required=True,
                    help='Source file path')
parser.add_argument('--to', dest='dest_filepath',
                    help='Destination file path. If not specified, it will be the same as the source file path, but with the opposite extension (.tsv or .pickle).')
parser.add_argument('--score_type', choices=['int', 'float'], default='float',
                    help='Type of destination (prediction or label) scores')
parser.add_argument('--id_type', choices=['int', 'str'], default='int',
                    help='Type of destination query or document IDs')
parser.add_argument('--is_qrels', action='store_true',
                    help="If the destination 'tsv' file is meant to be in a qrels format (including a 'Q0' column and no rank column).")
parser.add_argument('--num_top', type=int, default=None,
                    help="If set, will write only the `num_top` documents for each query (in descending order of score).")
args = parser.parse_args()


def convert_tsv_to_pickle(input_path, output_path, score_type, id_type, num_top=None):
    """
    Reads a file with scored candidate documents for each query into a 
    dictionary {qid: {passageid: relevance}} and writes it to a new binary file.
    
    :param input_path: input tsv file path
    :param output_path: output pickle file path
    :param score_type: Only for format 'tsv': the type of the written scores (int or float).
    :param id_type: type of query and passage IDs (int or str)
    """

    score_type = __builtins__.int if score_type =='int' else __builtins__.float
    id_type = __builtins__.int if id_type =='int' else __builtins__.str

    logger.info("Reading scores from {} ...".format(input_path))
    scores = utils.load_qrels(input_path, relevance_level=0, rel_type=score_type, id_type=id_type)
    
    if num_top is not None:
        logger.info("Keeping only the top {} passages for each query ...".format(num_top))
        for qid in scores:
            scores[qid] = dict(sorted(scores[qid].items(), key=itemgetter(1), reverse=True)[:num_top])
            # TODO: to remove boost factor from first passage score. RESTORE BY REMOVING BELOW!
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

    score_type = __builtins__.int if score_type =='int' else __builtins__.float
    # id_type = __builtins__.int if id_type =='int' else __builtins__.str

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
    
    orig_main_filename, orig_ext = os.path.splitext(args.orig_filepath)
    if orig_ext == '.pickle' or orig_ext == '.pkl':
        dest_type = '.tsv'
    else:
        dest_type = '.pickle'
    if args.dest_filepath is None:
        args.dest_filepath = orig_main_filename + dest_type

    if dest_type == '.pickle':
        convert_tsv_to_pickle(args.orig_filepath, args.dest_filepath, args.score_type, args.id_type, args.num_top)
    else:
        convert_pickle_to_tsv(args.orig_filepath, args.dest_filepath, args.score_type, args.is_qrels, args.num_top)
        
    logger.info("Done!")
