import json
import os
import glob
import argparse
import logging
from collections import OrderedDict

logger = logging.getLogger()

import utils


# METRICS = ['ndcg_cut_10', 'ndcg_cut_100', 'recip_rank', 'recall_10', 'recall_100', 'recall_1000']

parser = argparse.ArgumentParser("Evaluate a set of prediction (ranking) files according to pattern, and report and tabulate TREC metrics")
parser.add_argument("--pred_path", type=str, default="/users/gzerveas/data/gzerveas/RecipNN/predictions/MS_MARCO/*.tsv",
                    help="Glob pattern used to select prediction/ranking files for evaluation. Can also be a single file path.")
parser.add_argument("--qrels_path", type=str, default="/users/gzerveas/data/MS_MARCO/qrels",
                    help="""Path to qrels (ground truth relevance judgements) as needed by trec_eval""")
parser.add_argument("--auto_qrels", action='store_true',
                    help="If set, will use `qrels_path` as a prefix and append the trailing part of files matched by `pred_path` "
                    "(e.g. '.dev.small.tsv', hence automatically generating the corresponding qrel paths.")
parser.add_argument("--write_to_json", action='store_true',
                    help="If set, will write metrics to JSON files whose path will match `pred_path`, but with a different extension."
                    "(e.g. '.dev.small.tsv', hence automatically generating the corresponding qrel paths.")
parser.add_argument('--records_file', default='./records.xlsx', help="Excel file keeping records of all experiments. If 'None', will not export results to an excel sheet.")
args = parser.parse_args()


filepaths = glob.glob(args.pred_path)

logger.info(f"Will evaluate {len(filepaths)} file(s)")

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
        
    
    logger.info(f"Evaluating '{filename}' using '{qrels_filename}' ...")
    
    results = utils.load_predictions(filename)
    qrels = utils.load_qrels(qrels_filename)
    
    perf_metrics = utils.get_retrieval_metrics(results, qrels)
    
    if args.write_to_json:
        eval_filename = '.'.join(filename_parts[:-1]) + '.json'
        with open(eval_filename, 'w') as fp:
            json.dump(perf_metrics, fp)
    
    print(perf_metrics)
    
    perf_metrics['time'] = '-'
    
    WEIGHT_FUNC = '-'
    WEIGHT_FUNC_PARAM = '-'
    NORMALIZATION = '-'
    parameters = OrderedDict()
    parameters['sim_mixing_coef'] = '-'
    parameters['k'] = '-'
    parameters['trust_factor'] = '-'
    parameters['k_exp'] = '-'
    parameters['normalize'] = NORMALIZATION
    parameters['weight_func'] = WEIGHT_FUNC
    parameters['weight_func_param'] = WEIGHT_FUNC_PARAM
        
    # Export record metrics to a file accumulating records from all experiments
    if args.records_file != 'None':
        utils.register_record(args.records_file, '-', os.path.basename(filename), perf_metrics, parameters=parameters)