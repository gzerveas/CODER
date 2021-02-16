import logging
logging.basicConfig(format='%(asctime)s | %(name)-8s - %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()
logger.info("Loading packages ...")
import os
import sys
import random
import time
import argparse
import json
import traceback
from datetime import datetime
import string
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler
# from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers import BertConfig, BertTokenizer, RobertaTokenizer, BertModel, RobertaModel, get_polynomial_decay_schedule_with_warmup

# Package modules
from modeling import RepBERT_Train, MDSTransformer
from dataset import MYMARCO_Dataset
from optimizers import get_optimizer_class
import utils
import metrics

from dataset import lookup_times, sample_fetching_times, collation_times, retrieve_candidates_times, prep_docids_times

NEG_METRICS = {}  # metrics which are better when smaller
METRICS = ['MRR', 'MAP', 'Recall', 'nDCG']  # metrics which are better when higher
METRICS.extend(m + '@' for m in METRICS[:])

val_times = utils.Timer()  # stores measured validation times


def run_parse_args():
    parser = argparse.ArgumentParser(description='Run a complete training or evaluation. Optionally, a JSON configuration '
                                                 'file can be used, to overwrite command-line arguments.')
    ## Run from config file
    parser.add_argument('--config', dest='config_filepath',
                        help='Configuration .json file (optional, typically *instead* of command line arguments). '
                             'Overwrites existing command-line args!')
    ## Experiment
    parser.add_argument('--name', dest='experiment_name', type=str, default='',
                        help='A string identifier/name for the experiment to be run '
                             '- it will be appended to the output directory name, before the timestamp')
    parser.add_argument('--comment', type=str, default='', help='A comment/description of the experiment')
    parser.add_argument('--no_timestamp', action='store_true',
                        help='If set, a timestamp and random suffix will not be appended to the output directory name')
    parser.add_argument("--task", choices=["train", "dev", "eval"], required=True)
    parser.add_argument('--resume', action='store_true',
                        help='Used together with `load_model`. '
                             'If set, will load `start_step` and state of optimizer, scheduler besides model weights.')

    ## I/O
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Root output directory. Must exist. Time-stamped directories will be created inside.')
    parser.add_argument("--msmarco_dir", type=str, default="~/data/MS_MARCO",
                        help="Directory where qrels, queries files can be found")
    parser.add_argument("--train_candidates_path", type=str, default="~/data/MS_MARCO/BM25_top1000.in_qrels.train.tsv",
                        help="Text file of candidate (retrieved) documents/passages per query. This can be produced by e.g. Anserini")
    parser.add_argument("--eval_candidates_path", type=str, default="~/data/MS_MARCO/BM25_top1000.in_qrels.dev.tsv",
                        help="Text file of candidate (retrieved) documents/passages per query. This can be produced by e.g. Anserini")
    parser.add_argument("--embedding_memmap_dir", type=str, default="repbert/representations/doc_embedding",
                        help="Directory containing (num_docs_in_collection, doc_emb_dim) memmap array of document "
                             "embeddings and an accompanying (num_docs_in_collection,) memmap array of doc/passage IDs")
    parser.add_argument("--tokenized_dir", type=str, default="repbert/preprocessed",
                        help="Contains pre-tokenized/numerized queries in JSON files")
    parser.add_argument("--collection_memmap_dir", type=str, default="./data/collection_memmap", help="RepBERT only!")  # RepBERT only
    parser.add_argument('--records_file', default='./records.xls', help='Excel file keeping best records of all experiments')
    parser.add_argument('--load_model', dest='load_model_path', type=str, help='Path to pre-trained model.')
    # The following are currently used only if `model_type` is NOT 'repbert'
    parser.add_argument("--query_encoder_from", type=str, default="bert-base-uncased",
                        help="""A string used to initialize the query encoder weights and config object: 
                        can be either a pre-defined HuggingFace transformers string (e.g. "bert-base-uncased"), or
                        a path of a directory containing weights and config file""")
    parser.add_argument("--query_encoder_config", type=str, default=None,
                        help="""A string used to define the query encoder configuration (optional):
                        Used in case only the weights should be initialized by `query_encoder_from`. 
                        Can be either a pre-defined HuggingFace transformers string (e.g. "bert-base-uncased"), or
                        a path of a directory containing the config file, or directly the JSON config path.""")
    parser.add_argument("--tokenizer_from", type=str, default=None,
                        help="""Path to a directory containing a saved custom tokenizer (vocabulary and added tokens).
                        It is optional and used together with `query_encoder_type`.""")

    ## Dataset
    parser.add_argument('--train_limit_size', type=float, default=None,
                        help="Limit  dataset to specified smaller random sample, e.g. for debugging purposes. "
                             "If in [0,1], it will be interpreted as a proportion of the dataset, "
                             "otherwise as an integer absolute number of samples")
    parser.add_argument('--eval_limit_size', type=float, default=None,
                        help="Limit  dataset to specified smaller random sample, e.g. for debugging purposes. "
                             "If in [0,1], it will be interpreted as a proportion of the dataset, "
                             "otherwise as an integer absolute number of samples")
    parser.add_argument("--max_query_length", type=int, default=32)
    parser.add_argument("--max_doc_length", type=int, default=256)  # RepBERT only
    parser.add_argument('--num_candidates', type=int, default=None,
                        help="Number of document IDs to sample from all document IDs corresponding to a query and found"
                             " in `candidates_path` file. If None, all found document IDs will be used.")
    parser.add_argument('--num_inbatch_neg', type=int, default=0,
                        help="Number of negatives to randomly sample from other queries in the batch for training. "
                             "If 0, only documents in `candidates_path` will be used as negatives.")

    ## System
    parser.add_argument('--debug', action='store_true', help="Activate debug mode")
    parser.add_argument('--n_gpu', type=int, default=-1,
                        help="Number of GPUs. Default (-1): Use all available. 0: Use CPU only.")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--data_num_workers", default=0, type=int,
                        help="Number of processes feeding data to model. Default: main process only.")
    parser.add_argument("--num_keep", default=1, type=int, help="How many (latest) checkpoints to keep, besides the best.")
    parser.add_argument("--load_collection_to_memory", action='store_true',
                        help="If true, will load entire doc. embedding array as np.array to memory, instead of memmap! "
                        "Needs ~26GB for MSMARCO (~50GB project total), but is faster.")

    ## Training process
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int)
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps. The model parameters will be updated every this many batches")
    parser.add_argument("--validation_steps", type=int, default=3000,
                        help="Validate every this many training steps (i.e. param. updates); 0 for never.")
    parser.add_argument("--save_steps", type=int, default=2000,
                        help="Save checkpoint every this many training steps (i.e. param. updates); "
                             "0 for no periodic saving (save only at the end)")
    parser.add_argument("--logging_steps", type=int, default=200,
                        help="Log training information (tensorboard) every this many training steps; 0 for never")

    parser.add_argument("--num_epochs", default=1, type=int)
    parser.add_argument('--optimizer', choices={"AdamW", "RAdam"}, default="AdamW", help="Optimizer")
    parser.add_argument("--learning_rate", default=1e-4, type=float)  # 3e-6
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)  # 1e-8
    parser.add_argument("--warmup_steps", default=1000, type=int)  # 10000
    parser.add_argument("--final_lr_ratio", default=0.5, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--weight_decay", default=0.002, type=float)  # 0.01

    ## Evaluation
    parser.add_argument("--metrics_k", type=int, default=10, help="Evaluate metrics considering top k candidates")
    parser.add_argument('--key_metric', choices=METRICS, default='MRR', help='Metric used for defining best epoch')

    ## Model
    parser.add_argument("--model_type", type=str, choices=['repbert', 'mdstransformer'], default='mdstransformer',
                        help="""Type of the entire (end-to-end) information retrieval model""")
    parser.add_argument("--query_encoder_type", type=str, choices=['bert', 'roberta'], default='bert',
                        help="""Type of the model component used for encoding queries""")
    # The following refer to the transformer "decoder" (which processes an input sequence of document embeddings)
    parser.add_argument('--d_model', type=int, default=None,
                        help='Internal dimension of transformer decoder embeddings')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                        help='Dimension of dense feedforward part of transformer decoder layer')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of multi-headed attention heads for the transformer decoder')
    parser.add_argument('--num_layers', type=int, default=4,
                        help="Number of transformer decoder 'layers' (blocks)")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout applied to most transformer decoder layers')
    parser.add_argument('--pos_encoding', choices={'fixed', 'learnable', 'none'}, default='none',
                        help='What kind of positional encoding to use for the transformer decoder input sequence')
    parser.add_argument('--activation', choices={'relu', 'gelu'}, default='gelu',
                        help='Activation to be used in transformer decoder')
    parser.add_argument('--normalization_layer', choices={'BatchNorm', 'LayerNorm'}, default='BatchNorm',
                        help='Normalization layer to be used internally in the transformer decoder')

    args = parser.parse_args()

    # User can enter e.g. 'MRR@', indicating that they want to use the provided metrics_k for the key metric
    components = args.key_metric.split('@')
    if len(components) > 1:
        args.key_metric = components[0] + "@{}".format(args.metrics_k)

    if args.resume and (args.load_model_path is None):
        raise ValueError("You can only use option '--resume' when also specifying a model to load!")

    return args


def train(args, model, val_dataloader, tokenizer=None):
    """Prepare training dataset, train the model and handle results"""

    utils.set_seed(args)  # for reproducibility

    tb_writer = SummaryWriter(args.tensorboard_dir)

    train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    logger.info("Preparing {} dataset ...".format('train'))
    start_time = time.time()
    train_dataset = MYMARCO_Dataset('train', args.embedding_memmap_dir, args.tokenized_dir, args.train_candidates_path,
                                    qrels_path=args.msmarco_dir, tokenizer=tokenizer,
                                    max_query_length=args.max_query_length, num_candidates=args.num_candidates,
                                    limit_size=args.train_limit_size,
                                    load_collection_to_memory=args.load_collection_to_memory,
                                    emb_collection=val_dataloader.dataset.emb_collection)
    collate_fn = train_dataset.get_collate_func(num_inbatch_neg=args.num_inbatch_neg)
    logger.info("'train' data loaded in {:.3f} sec".format(time.time() - start_time))

    # NOTE RepBERT: Must be sequential! Pos, Neg, Pos, Neg, ...
    # This is because a (query, pos. doc, neg. doc) triplet is split in 2 consecutive samples: (qID, posID) and (qID, negID)
    # If random sampling had been chosen, then these 2 samples would have ended up in different batches
    # train_sampler = SequentialSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=train_batch_size, num_workers=args.data_num_workers,
                                  collate_fn=collate_fn)

    epoch_steps = (len(train_dataloader) // args.grad_accum_steps)  # num. actual steps (param. updates) per epoch
    total_training_steps = epoch_steps * args.num_epochs

    start_step = 0  # which step training started from

    # Prepare optimizer and schedule
    no_decay_str = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [param for name, param in model.named_parameters() if
                    not any(pattern in name for pattern in no_decay_str)],
         'weight_decay': args.weight_decay},
        {'params': [param for name, param in model.named_parameters() if any(nd in name for nd in no_decay_str)],
         'weight_decay': 0.0}
    ]
    optim_class = get_optimizer_class(args.optimizer)
    optimizer = optim_class(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                          num_training_steps=total_training_steps,
                                                          lr_end=args.final_lr_ratio*args.learning_rate, power=1.0)

    # Load model and possibly optimizer/scheduler state
    if args.load_model_path:
        model, start_step, optimizer, scheduler = utils.load_model(model, args.load_model_path, optimizer, scheduler,
                                                                   args.resume)
    model.to(args.device)
    utils.move_to_device(optimizer, args.device)

    global_step = start_step  # counts how many times the weights have been updated, i.e. num. batches // gradient acc. steps
    start_epoch = global_step // epoch_steps
    if start_step >= total_training_steps:
        logger.error("The loaded model has been already trained for {} steps ({} epochs), "
                     "while specified `num_epochs` is {} (total steps {})".format(start_epoch, start_step,
                                                                                  args.num_epochs, total_training_steps))
        sys.exit(1)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train
    logger.info("\n\n***** START TRAINING *****\n\n")
    logger.info("Number of epochs: %d", args.num_epochs)
    logger.info("Number of training examples: %d", len(train_dataset))
    logger.info("Number of validation examples: {}".format(len(val_dataloader.dataset)))
    logger.info("Batch size per GPU: %d", args.per_gpu_train_batch_size)
    logger.info("Total train batch size (w. parallel, distributed & accumulation): %d",
                train_batch_size * args.grad_accum_steps)
    logger.info("Gradient Accumulation steps: %d", args.grad_accum_steps)
    logger.info("Total optimization steps: %d", total_training_steps)

    best_value = 1e16 if args.key_metric in NEG_METRICS else -1e16  # initialize with +inf or -inf depending on key metric
    running_metrics = []  # (for validation) list of lists: for every evaluation, stores metrics like loss, MRR, MAP, ...
    best_metrics = {}

    train_loss = 0  # this is the training loss accumulated from the beginning of training
    logging_loss = 0  # this is synchronized with `train_loss` every args.logging_steps
    model.zero_grad()
    model.train()
    epoch_iterator = trange(start_epoch, int(args.num_epochs), desc="Epochs")

    batch_times = utils.Timer()  # average time for the model to train (forward + backward pass) on a single batch of queries
    for epoch_idx in epoch_iterator:
        epoch_start_time = time.time()
        batch_iterator = tqdm(train_dataloader, desc="Batches")
        for step, (model_inp, _, _) in enumerate(batch_iterator):  # step can be a "sub-step", if grad. accum. > 1
            if args.resume and ((epoch_idx * epoch_steps) + step < args.grad_accum_steps * global_step):
                continue  # this is done to continue dataloader from the correct step, when using args.resume

            model_inp = {k: v.to(args.device) for k, v in model_inp.items()}
            start_time = time.perf_counter()
            output = model(**model_inp)  # model output is a dictionary
            loss = output['loss']

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.grad_accum_steps > 1:
                loss = loss / args.grad_accum_steps
            loss.backward()  # calculate gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            train_loss += loss.item()

            batch_times.update(time.perf_counter() - start_time)

            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # logging for training
                if args.logging_steps and (global_step % args.logging_steps == 0):
                    tb_writer.add_scalar('learn_rate_g0', scheduler.get_last_lr()[0], global_step)  # [0] for first group
                    tb_writer.add_scalar('learn_rate_g1', scheduler.get_last_lr()[1], global_step)  # [1] for second group
                    cur_loss = (train_loss - logging_loss) / args.logging_steps  # mean loss over last args.logging_steps (smoothened "current loss")
                    tb_writer.add_scalar('train/loss', cur_loss, global_step)

                    if args.debug:
                        logger.debug("Mean loss over {} steps: {:.5f}".format(args.logging_steps, cur_loss))
                        logger.debug("Learning rate: {}".format(scheduler.get_last_lr()))
                        logger.debug("Current memory usage: {} MB or {} MB".format(np.round(utils.get_current_memory_usage()),
                                                                                   np.round(utils.get_current_memory_usage2())))
                        logger.debug("Max memory usage: {} MB".format(int(np.ceil(utils.get_max_memory_usage()))))

                        logger.debug("Average lookup time: {} s /samp".format(lookup_times.get_average()))
                        logger.debug("Average retr. candidates time: {} s /samp".format(retrieve_candidates_times.get_average()))
                        logger.debug("Average prep. docids time: {} s /samp".format(prep_docids_times.get_average()))
                        logger.debug("Average sample fetching time: {} s /samp".format(sample_fetching_times.get_average()))
                        logger.debug("Average collation time: {} s /batch".format(collation_times.get_average()))
                        logger.debug("Average total batch processing time: {} s /batch".format(batch_times.get_average()))

                    logging_loss = train_loss

                # evaluate at specified interval or if this is the last step
                if (args.validation_steps and (global_step % args.validation_steps == 0)) or global_step == total_training_steps:

                    logger.info("\n\n***** Running evaluation of step {} on dev set *****".format(global_step))
                    val_metrics, best_metrics, best_value = validate(args, model, val_dataloader, tb_writer,
                                                                     best_metrics, best_value, global_step)
                    metrics_names, metrics_values = zip(*val_metrics.items())
                    running_metrics.append(list(metrics_values))

                if (args.save_steps and (global_step % args.save_steps == 0)) or global_step == total_training_steps:
                    # Save model checkpoint
                    utils.remove_oldest_checkpoint(args.save_dir, args.num_keep)
                    utils.save_model(os.path.join(args.save_dir, 'model_{}.pth'.format(global_step)),
                                     global_step, model, optimizer, scheduler)
        epoch_runtime = time.time() - epoch_start_time
        logger.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(epoch_runtime)))

    # Export evolution of metrics over epochs
    header = metrics_names
    metrics_filepath = os.path.join(args.output_dir, "metrics_" + args.experiment_name + ".xls")
    book = utils.export_performance_metrics(metrics_filepath, running_metrics, header, sheet_name="metrics")

    # Export record metrics to a file accumulating records from all experiments
    utils.register_record(args.records_file, args.initial_timestamp, args.experiment_name,
                          best_metrics, val_metrics, comment=args.comment)

    avg_batch_time = batch_times.total_time / (epoch_idx * epoch_steps + step + 1)
    logger.info("Average time to train on 1 batch ({} samples): {:.6f} sec"
                " ({:.6f}s per sample)".format(train_batch_size, avg_batch_time, avg_batch_time/train_batch_size))
    logger.info("Average time to train on 1 batch ({} samples): {:.6f} sec".format(train_batch_size, batch_times.get_average()))
    logger.info('Best {} was {}. Other metrics: {}'.format(args.key_metric, best_value, best_metrics))

    logger.debug("Average lookup time: {} s".format(lookup_times.get_average()))
    logger.debug("Average retr. candidates time: {} s".format(retrieve_candidates_times.get_average()))
    logger.debug("Average prep. docids time: {} s".format(prep_docids_times.get_average()))
    logger.debug("Average sample fetching time: {} s".format(sample_fetching_times.get_average()))
    logger.debug("Average collation time: {} s".format(collation_times.get_average()))

    return


def validate(args, model, val_dataloader, tensorboard_writer, best_metrics, best_value, global_step):
    """Run an evaluation on the validation set while logging metrics, and handle result"""

    model.eval()
    eval_start_time = time.time()
    val_metrics, ranked_df = evaluate(args, model, val_dataloader)
    eval_runtime = time.time() - eval_start_time
    model.train()
    logger.info("Validation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))

    global val_times
    val_times.update(eval_runtime)
    avg_val_time = val_times.get_average()
    avg_val_batch_time = avg_val_time / len(val_dataloader)
    avg_val_sample_time = avg_val_time / len(val_dataloader.dataset)
    logger.info("Avg val. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_val_time)))
    logger.info("Avg batch val. time: {} seconds".format(avg_val_batch_time))
    logger.info("Avg sample val. time: {} seconds".format(avg_val_sample_time))

    print_str = 'Step {} Validation Summary: '.format(global_step)
    for k, v in val_metrics.items():
        tensorboard_writer.add_scalar('dev/{}'.format(k), v, global_step)
        print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)

    val_metrics["global_step"] = global_step
    if args.key_metric in NEG_METRICS:
        condition = (val_metrics[args.key_metric] < best_value)
    else:
        condition = (val_metrics[args.key_metric] > best_value)
    if condition:
        best_value = val_metrics[args.key_metric]
        utils.save_model(os.path.join(args.save_dir, 'model_best.pth'), global_step, model)
        best_metrics = val_metrics.copy()

        ranked_filepath = os.path.join(args.pred_dir, 'best.ranked.dev.tsv')
        ranked_df.to_csv(ranked_filepath, header=False, sep='\t')

        # Export metrics to a file accumulating best records from the current experiment
        rec_filepath = os.path.join(args.pred_dir, 'training_session_records.xls')
        utils.register_record(rec_filepath, args.initial_timestamp, args.experiment_name, best_metrics)

    return val_metrics, best_metrics, best_value


def evaluate(args, model, dataloader):
    """
    Evaluate a given model on the dataset contained in the given dataloader and compile a dataframe with
    document ranks and scores for each query. If the dataset includes relevance labels (qrels), then metrics
    such as MRR, MAP etc will be additionally computed.
    :return:
        eval_metrics: dict containing metrics (at least 1, batch processing time)
        rank_df: dataframe with indexed by qID (shared by multiple rows) and columns: PID, rank, score
    """
    qrels = dataloader.dataset.qrels
    labels_exist = qrels is not None

    # num_docs is the (potentially variable) number of candidates per query
    relevances = []  # (total_num_queries) list of (num_docs) lists with 1 at the indices corresponding to actually relevant passages
    num_relevant = []  # (total_num_queries) list of number of ground truth relevant documents per query
    df_chunks = []  # (total_num_queries) list of dataframes, each with index a single qID and corresponding (num_docs) columns PID, rank, score
    query_time = 0  # average time for the model to score candidates for a single query
    total_loss = 0  # total loss over dataset

    with torch.no_grad():
        for batch_data, qids, docids in tqdm(dataloader, desc="Evaluating"):
            batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
            start_time = time.perf_counter()
            out = model(**batch_data)  # (batch_size, num_docs) relevance scores in [0, 1], because not 'train'
            query_time += time.perf_counter() - start_time
            rel_scores = out['rel_scores'].detach().cpu().numpy()
            if 'loss' in out:
                total_loss += out['loss'].item()
            assert len(qids) == len(docids) == len(rel_scores)

            # Rank documents based on their scores
            num_docs_per_query = [len(cands) for cands in docids]
            num_lengths = set(num_docs_per_query)
            no_padding = (len(num_lengths) == 1)  # all queries in this batch had the same number of candidates

            if no_padding:  # (only) 10% speedup compared to other case
                sort_inds = np.fliplr(np.argsort(rel_scores, axis=1))  # (batch_size, num_docs) inds to sort rel_scores
                # (batch_size, num_docs) docIDs per query, in order of descending relevance score
                ranksorted_docs = np.take_along_axis(np.array(docids, dtype=np.int32), sort_inds, axis=1)
                sorted_scores = np.take_along_axis(rel_scores, sort_inds, axis=1)
            else:
                # (batch_size) iterables of docIDs and scores per query, in order of descending relevance score
                ranksorted_docs, sorted_scores = zip(*(map(rank_docs, docids, rel_scores)))

            # extend by batch_size elements
            df_chunks.extend(pd.DataFrame(data={"PID": ranksorted_docs[i],
                                                "rank": list(range(len(docids[i]))),
                                                "score": sorted_scores[i]},
                                          index=[qids[i]]*len(docids[i])) for i in range(len(qids)))
            if labels_exist:
                relevances.extend(get_relevances(qrels[qids[i]], ranksorted_docs[i]) for i in range(len(qids)))
                num_relevant.extend(len(qrels[qid]) for qid in qids)

    if labels_exist:
        try:
            eval_metrics = calculate_metrics(relevances, num_relevant, args.metrics_k)  # aggr. metrics for the entire dataset
        except:
            logger.error('Metrics calculation failed!')
            eval_metrics = OrderedDict()
        eval_metrics['loss'] = total_loss / len(dataloader.dataset)  # average over samples
    else:
        eval_metrics = OrderedDict()
    eval_metrics['query_time'] = query_time / len(dataloader.dataset)  # average over samples
    ranked_df = pd.concat(df_chunks, copy=False)  # index: qID (shared by multiple rows), columns: PID, rank, score

    return eval_metrics, ranked_df


def rank_docs(docids, scores):
    """Given a list of document IDs and a (potentially longer due to padding) 1D array of their scores, sort both scores
    and coresponding document IDs in the order of descending scores."""
    actual_scores = scores[:len(docids)]
    sort_inds = np.flip(np.argsort(actual_scores))
    actual_scores = actual_scores[sort_inds]
    docids = [docids[i] for i in sort_inds]
    return docids, actual_scores


def get_relevances(gt_relevant, candidates, max_docs=None):
    """Assumes a *single* level of relevance (1), an assumption that holds for MSMARCO qrels.{train, dev}.tsv
    Args: # TODO: can be easily extended to account for multiple levels of relevance in ground truth labels
        gt_relevant: set of ground-truth relevant passage IDs corresponding to a single query
        candidates: list of candidate pids
        max_docs: consider only the first this many documents
    Returns: list of length min(max_docs, len(pred)) with 1 at the indices corresponding to passages in `gt_relevant`
        e.g. [0 1 1 0 0 1 0]
    """
    if max_docs is None:
        max_docs = len(candidates)
    return [1 if pid in gt_relevant else 0 for pid in candidates[:max_docs]]


def calculate_metrics(relevances, num_relevant, k):

    eval_metrics = OrderedDict([('MRR@{}'.format(k), metrics.mean_reciprocal_rank(relevances, k)),
                                ('MAP@{}'.format(k), metrics.mean_average_precision(relevances, k)),
                                ('Recall@{}'.format(k), metrics.recall_at_k(relevances, num_relevant, k)),
                                ('nDCG@{}'.format(k), np.mean([metrics.ndcg_at_k(rel, k) for rel in relevances])),

                                ('MRR', metrics.mean_reciprocal_rank(relevances)),
                                ('MAP', metrics.mean_average_precision(relevances)),
                                ('nDCG', np.mean([metrics.ndcg_at_k(rel, k) for rel in relevances]))])
    return eval_metrics


# Very inefficient. Used by RepBERT only.
def evaluate_slow(args, model, dataloader, mode, prefix):
    """
    Writes scores to file while evaluating, then reads file, sorts results and writes another file with ranks, then runs
    MSMARCO evaluation Python script on this file (again reads many things) and uses regex to get MRR metric from output
    """

    logger.info("***** Running evaluation of {} on {} set *****".format(prefix, mode))

    output_file_path = os.path.join(args.pred_dir, prefix + ".{}.score.tsv".format(mode))
    with open(output_file_path, 'w') as outputfile:
        with torch.no_grad():
            for batch, qids, docids in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(args.device) for k, v in batch.items()}
                outputs = model(**batch)  # Tuple(similarities, query_embeddings, doc_embeddings)
                # outputs[0] is a (batch_size, batch_size) tensor of similarities between each query in the batch and each document in the batch
                scores = torch.diagonal(outputs[0]).detach().cpu().numpy()
                assert len(qids) == len(docids) == len(scores)
                for qid, docid, score in zip(qids, docids, scores):
                    outputfile.write(f"{qid}\t{docid}\t{score}\n")

    rank_output = os.path.join(args.pred_dir, prefix + ".{}.rank.tsv".format(mode))
    utils.generate_rank(output_file_path, rank_output)

    if mode == "dev":
        mrr = utils.eval_results(rank_output)
        return mrr


def main():
    args = run_parse_args()
    if args.debug:
        logger.setLevel('DEBUG')

    # Setup experiment session, convert config dict to args object
    args = utils.dict2obj(setup(args))

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run

    # Setup CUDA, GPU 
    args.device = torch.device("cuda" if torch.cuda.is_available() and (args.n_gpu != 0) else "cpu")
    if args.n_gpu < 0:
        args.n_gpu = torch.cuda.device_count()
    elif args.device.type == 'cpu':
        args.n_gpu = 0

    # Log current hardware setup
    logger.info("Device: %s, n_gpu: %s", args.device, args.n_gpu)

    # Set seed
    utils.set_seed(args)

    # Get tokenizer
    tokenizer = get_tokenizer(args)

    # Load evaluation set and initialize evaluation dataloader
    if args.task == 'train':
        eval_mode = 'dev'  # 'eval' here is the name of the MSMARCO test set, 'dev' is the validation set
    else:
        eval_mode = args.task

    logger.info("Preparing {} dataset ...".format(eval_mode))
    start_time = time.time()
    eval_dataset = MYMARCO_Dataset(eval_mode, args.embedding_memmap_dir, args.tokenized_dir, args.eval_candidates_path,
                                   qrels_path=args.msmarco_dir, tokenizer=tokenizer,
                                   max_query_length=args.max_query_length, num_candidates=None,
                                   limit_size=args.eval_limit_size,
                                   load_collection_to_memory=args.load_collection_to_memory)
    collate_fn = eval_dataset.get_collate_func()
    logger.info("'{}' data loaded in {:.3f} sec".format(eval_mode, time.time() - start_time))

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                 num_workers=args.data_num_workers, collate_fn=collate_fn)

    logger.info("Number of {} samples: {}".format(eval_mode, len(eval_dataset)))
    logger.info("Batch size: %d", args.eval_batch_size)

    # Initialize model. This is done after loading the data, to know the doc. embeddings dimension
    logger.info("Initializing model ...")
    if args.model_type == 'repbert':
        # keep configuration setup like RepBERT (for backward compatibility).
        # The model is a common/shared BERT query-document encoder, without interactions between query and document token representations
        if args.load_model_path is None:
            args.load_model_path = "bert-base-uncased"
        # Works with either directory path containing HF config file, or JSON HF config file,  or pre-defined model string
        config = BertConfig.from_pretrained(args.load_model_path)
        model = RepBERT_Train.from_pretrained(args.load_model_path, config=config)
    else:  # new configuration setup for MultiDocumentScoringTransformer models
        model = get_model(args, eval_dataset.emb_collection.embedding_vectors.shape[1])

    logger.debug("Model:\n{}".format(model))
    logger.info("Total number of parameters: {}".format(utils.count_parameters(model)))
    logger.info("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))

    if args.task == "train":
        train(args, model, eval_dataloader, tokenizer)
    else:
        # Just evaluate trained model on some dataset (needs ~27GB for MS MARCO dev set)

        # only composite (non-repbert) models need to be loaded; repbert is already loaded at this point
        if args.model_type != 'repbert':
            model, global_step, _, _ = utils.load_model(model, args.load_model_path)
        model.to(args.device)
        model_checkpoint_name = os.path.splitext(os.path.basename(args.load_model_path))[0]
        logger.info("Will evaluate model on {} set".format(args.load_model_path, args.task))

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        model.eval()
        eval_start_time = time.time()
        eval_metrics, ranked_df = evaluate(args, model, eval_dataloader)
        eval_runtime = time.time() - eval_start_time
        logger.info("Evaluation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))
        print()
        print_str = 'Evaluation Summary: '
        for k, v in eval_metrics.items():
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)

        ranked_filepath = os.path.join(args.pred_dir, 'ranked.eval.tsv')
        logger.info("Writing predicted ranking to: {} ...".format(ranked_filepath))
        ranked_df.to_csv(ranked_filepath, header=False, sep='\t')


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    config = args.__dict__  # configuration dictionary

    if args.config_filepath is not None:
        logger.info("Reading configuration ...")
        try:  # dictionary containing the entire configuration settings in a hierarchical fashion
            config.update(utils.load_config(args.config_filepath))
        except:
            logger.critical("Failed to load configuration file. Check JSON syntax and verify that files exist")
            traceback.print_exc()
            sys.exit(1)

    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(output_dir))

    output_dir = os.path.join(output_dir, config['experiment_name'])

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    config['initial_timestamp'] = formatted_timestamp
    if (not config['no_timestamp']) or (len(config['experiment_name']) == 0):
        rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
        output_dir += "_" + formatted_timestamp + "_" + rand_suffix
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    utils.create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config


def get_query_encoder(args):
    """Initialize and return query encoder model object based on args"""

    if args.query_encoder_type == 'bert':
        return BertModel.from_pretrained(args.query_encoder_from, config=args.query_encoder_config)
    elif args.query_encoder_type == 'roberta':
        return RobertaModel.from_pretrained(args.query_encoder_from, config=args.query_encoder_config)


def get_model(args, doc_emb_dim=None):
    """Initialize and return end-to-end model object based on args"""

    query_encoder = get_query_encoder(args)

    if args.model_type == 'mdstransformer':

        return MDSTransformer(custom_encoder=query_encoder,
                              d_model=args.d_model,
                              num_heads=args.num_heads,
                              num_decoder_layers=args.num_layers,
                              dim_feedforward=args.dim_feedforward,
                              dropout=args.dropout,
                              activation=args.activation,
                              doc_emb_dim=doc_emb_dim)
    else:
        raise NotImplementedError('Unknown model type')


def get_tokenizer(args):
    """Initialize and return tokenizer object based on args"""

    if args.query_encoder_type == 'bert':
        if args.tokenizer_from is None:  # if not a directory path
            args.tokenizer_from = 'bert-base-uncased'
        return BertTokenizer.from_pretrained(args.tokenizer_from)
    elif args.query_encoder_type == 'roberta':
        if args.tokenizer_from is None:  # if not a directory path
            args.tokenizer_from = 'roberta-base'
        return RobertaTokenizer.from_pretrained(args.tokenizer_from)


if __name__ == "__main__":
    total_start_time = time.time()
    main()
    logger.info("All done!")
    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))
