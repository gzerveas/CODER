import os
import glob
import argparse
import logging
from collections import defaultdict, OrderedDict

logger = logging.getLogger()

import utils


# METRICS = ['ndcg_cut_10', 'ndcg_cut_100', 'recip_rank', 'recall_10', 'recall_100', 'recall_1000']

parser = argparse.ArgumentParser("Evaluate a set of prediction (ranking) files according to pattern, and report and tabulate TREC metrics")

# parser.add_argument("--trec_eval_binary", type=str, default='/users/gzerveas//data/MS_MARCO/trec2019/trec_eval-9.0.7/trec_eval',
#                     help="If given, the binary file of trec_eval will be executed to evaluate ranking performnance")
parser.add_argument("--pred_path", type=str, default="/users/gzerveas/data/gzerveas/RecipNN/predictions/MS_MARCO/*.tsv",
                    help="Glob pattern used to select prediction/ranking files for evaluation. Can also be a single file path.")
parser.add_argument("--qrels_path", type=str, default="/users/gzerveas/data/MS_MARCO/qrels",
                    help="""Path to qrels (ground truth relevance judgements) as needed by trec_eval""")
parser.add_argument('--records_file', default='./records.xls', help='Excel file keeping records of all experiments')
args = parser.parse_args()


filepaths = glob.glob(args.pred_path)

logger.info(f"Will evaluate {len(filepaths)} file(s)")

for filename in filepaths:
    filename_parts = filename.split('.')  # e.g. /my/pa.th/method_par0.444_descr.dev.small.tsv
    num_trailing_parts = 0  # e.g. ['dev', 'small', 'tsv']
    for i in range(len(filename_parts)-1, 0, -1):
        if ('_' in filename_parts[i]) or ('/' in filename_parts[i]): # indicator that we are not in the trailing parts
            break
        else:
            num_trailing_parts += 1
    trailing_part = '.'.join(filename_parts[-num_trailing_parts:])  # e.g. "dev.small.tsv"
        
    qrels_filename = args.qrels_path + '.' + trailing_part
    eval_filename = '.'.join(filename_parts[:-1]) + '.eval'
    
    logger.info(f"Evaluating '{eval_filename}' using '{qrels_filename}' ...")
    
    results = utils.load_predictions(filename)
    qrels = utils.load_qrels(qrels_filename)
    
    perf_metrics = utils.get_retrieval_metrics(results, qrels)
    print(perf_metrics)
    
    WEIGHT_FUNC = 'exp'
    WEIGHT_FUNC_PARAM = 2.4
    NORMALIZATION = 'max'
    parameters = OrderedDict()
    parameters['sim_mixing_coef'] = '-'
    parameters['k'] = 20
    parameters['trust_factor'] = '-'
    parameters['k_exp'] = 6
    parameters['normalize'] = NORMALIZATION
    parameters['weight_func'] = WEIGHT_FUNC
    parameters['weight_func_param'] = WEIGHT_FUNC_PARAM
        
    # Export record metrics to a file accumulating records from all experiments
    utils.register_record(args.records_file, '-', os.path.basename(filename), perf_metrics, parameters=parameters)