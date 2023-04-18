import logging
logger = logging.getLogger()
logger.info("Loading packages...")

import json
import os
import glob
import argparse
import logging
from collections import OrderedDict
import pickle

import utils


# METRICS = ['ndcg_cut_10', 'ndcg_cut_100', 'recip_rank', 'recall_10', 'recall_100', 'recall_1000']

parser = argparse.ArgumentParser("Evaluate a set of prediction/ranking files (in '.tsv' or '.pickle' format) according to pattern, and report and tabulate TREC metrics")
parser.add_argument("--pred_path", type=str, default="/users/gzerveas/data/gzerveas/RecipNN/predictions/MS_MARCO/*.tsv",
                    help="Glob pattern used to select prediction/ranking files for evaluation (remember to enclose in quotes!). Can also be a single file path.")
parser.add_argument("--qrels_path", type=str, default="/users/gzerveas/data/MS_MARCO/qrels",
                    help="""Path to qrels (ground truth relevance judgements) as needed by trec_eval""")
parser.add_argument("--auto_qrels", action='store_true',
                    help="If set, will use `qrels_path` as a prefix and append the trailing part of files matched by `pred_path` "
                    "(e.g. '.dev.small.tsv', hence automatically generating the corresponding qrel paths.")
parser.add_argument("--relevance_level", type=int, default=1,
                    help="A document with this score level and above will be considered relevant.")
parser.add_argument("--write_to_json", action='store_true',
                    help="If set, will write metrics to JSON files whose path will match `pred_path`, but with a different extension."
                    "(e.g. '.dev.small.tsv', hence automatically generating the corresponding qrel paths.")
parser.add_argument('--records_file', default='./records.xls', help="Excel file keeping records of all experiments. If 'None', will not export results to an excel sheet.")
parser.add_argument("--parameters", default=None, type=str,
                    help="By default, no experiment parameters will be exported to the excel sheet, only metrics. "
                         "If set to 'empty', it will create parameters columns, which will be empty spaceholders. "
                         "Otherwise, it should be a string of the form {\"param\": value} that can be parsed to a dictionary.")
args = parser.parse_args()


filepaths = glob.glob(args.pred_path)

if len(filepaths) == 0:
    raise ValueError(f"No files found for pattern: {args.pred_path}")
logger.info(f"Will evaluate {len(filepaths)} file(s)")

old_qrels_filename = None
for filename in filepaths:
    filename_parts = filename.split('.')  # e.g. /my/pa.th/method_par0.444_descr.dev.small.tsv
    qrels_filename = args.qrels_path
    if args.auto_qrels:
        num_trailing_parts = 0  # e.g. ['dev', 'small', 'tsv']
        for i in range(len(filename_parts)-1, 0, -1):
            if ('_' in filename_parts[i]) or ('/' in filename_parts[i]): # indicator that we are not in the trailing parts
                break
            else:
                num_trailing_parts += 1
        trailing_part = '.'.join(filename_parts[-num_trailing_parts:])  # e.g. "dev.small.tsv"
        
        qrels_filename += '.' + trailing_part
    
    logger.info(f"Evaluating '{filename}' using '{qrels_filename}' :")
    
    if qrels_filename != old_qrels_filename:
        logger.info(f"Loading qrels from '{qrels_filename}' ...")
        qrels = utils.load_qrels(qrels_filename, relevance_level=args.relevance_level)
        old_qrels_filename = qrels_filename    
    
    logger.info("Reading scores from {} ...".format(filename))
    if filename.endswith('.tsv'):
        results = utils.load_predictions(filename)
    elif filename.endswith('.pickle'):    
        with open(filename, 'rb') as f:
            results = pickle.load(f)  # dict{qID: dict{pID: relevance}}
    else:
        raise ValueError(f"Unknown file format: {filename}")
    
    perf_metrics = utils.get_retrieval_metrics(results, qrels)
    
    if args.write_to_json:
        eval_filename = '.'.join(filename_parts[:-1]) + '.json'
        with open(eval_filename, 'w') as fp:
            json.dump(perf_metrics, fp)
    
    print(perf_metrics)
    
    if args.parameters == 'empty':  # parameter columns will be created but empty
        perf_metrics['time'] = ''
        
        WEIGHT_FUNC = ''
        WEIGHT_FUNC_PARAM = ''
        NORMALIZATION = ''
        parameters = OrderedDict()
        parameters['sim_mixing_coef'] = ''
        parameters['k'] = ''
        parameters['trust_factor'] = ''
        parameters['k_exp'] = ''
        parameters['normalize'] = NORMALIZATION
        parameters['weight_func'] = WEIGHT_FUNC
        parameters['weight_func_param'] = WEIGHT_FUNC_PARAM
    elif args.parameters is None:
        parameters = None # do not export parameters; columns will be missing
    else:
        try:
            parameters = eval(args.parameters)  # parse string to dict
        except:
            logger.error(f"Could not parse parameters dictionary from string: {args.parameters}")
            parameters = None
        
    # Export record metrics to a file accumulating records from all experiments
    if args.records_file != 'None':
        utils.register_record(args.records_file, '', os.path.basename(filename), perf_metrics, parameters=parameters)