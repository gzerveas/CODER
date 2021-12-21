import json
import time
import os
import pdb
import sys
from datetime import datetime
import subprocess
import argparse

def run_command(command):
    print (command)
    #stdoutdata = subprocess.getoutput(command)
    #return stdoutdata.split('\n')
    return

parser = argparse.ArgumentParser()
parser.add_argument('--jsonpath', type=str)
parser.add_argument('--gpuid', action='store', type=str, default="0",
                    help="cuda device ids for single/multi gpu setting like '0' or '0,1,2,3' ")
parser.add_argument('--expname', type=str, default="DEBUG")
parser.add_argument("--per_gpu_eval_batch_size", default=-1, type=int)
parser.add_argument("--per_gpu_train_batch_size", default=-1, type=int)



args = parser.parse_args()



command = "python main.py"

params_init = {'task': 'train', 
               'output_dir': '/share/home/navid/experiments/neuralir/msmarco-passage/multidocscoring',
               'embedding_memmap_dir': '/share/cp/datasets/ir/msmarco/passage/repbert/representations/doc_embedding',
               'tokenized_path': '/share/cp/datasets/ir/msmarco/passage/repbert/preprocessed',
               'qrels_path': '/share/cp/datasets/ir/msmarco/passage/qrels.dev.tsv',
               'train_candidates_path': '/share/cp/datasets/ir/msmarco/passage/repbert/preprocessed/BM25_top1000.in_qrels.train_memmap',
               'eval_candidates_path': '/share/cp/datasets/ir/msmarco/passage/repbert/preprocessed/BM25_top1000.in_qrels.dev_memmap',
               'records_file': 'MDST_records.xls'
              }
if args.per_gpu_eval_batch_size != -1:
    params_init['per_gpu_eval_batch_size'] = args.per_gpu_eval_batch_size
if args.per_gpu_train_batch_size != -1:
    params_init['per_gpu_train_batch_size'] = args.per_gpu_train_batch_size


command += " --gpu-id '%s'" % args.gpuid
command += " --name %s" % args.expname 

black_list = set(['initial_timestamp', 'experiment_name', 'n_gpu'])

with open(args.jsonpath, 'r') as fr:
    params_loaded = json.loads(fr.read())

for k, v in params_init.items():
    command += " --%s %s" % (k, v)

for k, v in params_loaded.items():
    if (k not in params_init) and (k not in black_list) and (v != None) and (v is not None):
        if type(v) == str:
            if v != '':
                command += " --%s '%s'" % (k, v)
        elif type(v) == bool:
            if v == True:
                command += " --%s" % (k)
        else:
            command += " --%s %s" % (k, v)
    
#pdb.set_trace()

run_command(command)

#--data_num_workers 0  --gpu-id 0 --train_limit_size 256 --logging_steps 2 --num_candidates 30 --num_random_neg 30 --load_collection_to_memory"

