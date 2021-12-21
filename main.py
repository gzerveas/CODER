import logging

import options

logging.basicConfig(format='%(asctime)s | %(name)-8s - %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()
logger.info("Loading packages ...")
import os
import sys
import random
import time
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
from transformers import BertConfig, BertTokenizer, RobertaTokenizer, BertModel, RobertaModel, AutoTokenizer, AutoModel

# Package modules
from options import *
from modeling import RepBERT_Train, MDSTransformer, get_loss_module, get_aggregation_function
from dataset import MYMARCO_Dataset, MSMARCODataset, load_original_sequences, EmbeddedSequences
from dataset import lookup_times, sample_fetching_times, collation_times, retrieve_candidates_times, prep_docids_times
from optimizers import get_optimizers, MultiOptimizer, get_schedulers, MultiScheduler
import utils
import metrics
from precompute import MSMARCO_DocDataset

from fair_retrieval.metrics_FaiRR import FaiRRMetric, FaiRRMetricHelper

val_times = utils.Timer()  # stores measured validation times


def train(args, model, val_dataloader, tokenizer=None, fairrmetric=None):
    """Prepare training dataset, train the model and handle results"""

    tb_writer = SummaryWriter(args.tensorboard_dir)

    train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    logger.info("Preparing {} dataset ...".format('train'))
    start_time = time.time()
    train_dataset = MYMARCO_Dataset('train', args.embedding_memmap_dir, args.tokenized_path, args.train_candidates_path,
                                    qrels_path=args.qrels_path, tokenizer=tokenizer,
                                    max_query_length=args.max_query_length, num_candidates=args.num_candidates,
                                    limit_size=args.train_limit_size,
                                    load_collection_to_memory=args.load_collection_to_memory,
                                    emb_collection=val_dataloader.dataset.emb_collection,
                                    collection_neutrality_path=args.collection_neutrality_path,
                                    query_ids_path=args.train_query_ids)
    collate_fn = train_dataset.get_collate_func(num_inbatch_neg=args.num_inbatch_neg, n_gpu=args.n_gpu)
    logger.info("'train' data loaded in {:.3f} sec".format(time.time() - start_time))

    utils.write_list(os.path.join(args.output_dir, "train_IDs.txt"), train_dataset.qids)
    utils.write_list(os.path.join(args.output_dir, "val_IDs.txt"), val_dataloader.dataset.qids)

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

    # Load model and possibly optimizer/scheduler state
    optim_state, sched_state = None, None
    if args.load_model_path:  # model is already on its intended device, which we pass as an argument
        model, start_step, optim_state, sched_state = utils.load_model(model, args.load_model_path, args.device, args.resume)

    # Prepare optimizer and schedule
    sstr = ['non-encoder', 'encoder']
    nonencoder_optimizer, encoder_optimizer = get_optimizers(args, model)
    logger.debug("args.learning_rate: {}".format(args.learning_rate))
    logger.debug('nonencoder_optimizer.defaults["lr"]: {}'.format(nonencoder_optimizer.defaults["lr"]))
    optimizer = MultiOptimizer(nonencoder_optimizer)
    if args.encoder_delay <= start_step:
        optimizer.add_optimizer(encoder_optimizer)
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)
        # TODO: Moving doesn't seem necessary
        # utils.move_to_device(nonencoder_optimizer, args.device)
        # utils.move_to_device(encoder_optimizer, args.device)
        logger.info('Loaded optimizer(s) state')
        logger.debug('optimizer.defaults["lr"]: {}'.format(optimizer.optimizers[0].defaults["lr"]))

    schedulers = get_schedulers(args, total_training_steps, nonencoder_optimizer, encoder_optimizer)

    logger.debug("schedulers['nonencoder_scheduler'].get_last_lr(): {}".format(schedulers['nonencoder_scheduler'].get_last_lr()))
    scheduler = MultiScheduler(schedulers['nonencoder_scheduler'])
    if args.reduce_on_plateau:
        ROP_scheduler = MultiScheduler(schedulers['ROP_nonencoder_scheduler'])
    if args.encoder_delay <= start_step:
        scheduler.add_scheduler(schedulers['encoder_scheduler'])
        if args.reduce_on_plateau:
            ROP_scheduler.add_scheduler(schedulers['ROP_encoder_scheduler'])
    if sched_state is not None:
        scheduler.load_state_dict(sched_state)
        logger.info('Loaded scheduler(s) state')
    scheduler.step()  # this is done to correctly initialize learning rate (otherwise optimizer.defaults["lr"] is the first value when using get_constant_schedule_with_warmup)
    logger.debug("schedulers['nonencoder_scheduler'].get_last_lr(): {}".format(scheduler.schedulers[0].get_last_lr()))

    global_step = start_step  # counts how many times the weights have been updated, i.e. num. batches // gradient acc. steps
    start_epoch = global_step // epoch_steps
    if start_step >= total_training_steps:
        logger.error("The loaded model has been already trained for {} steps ({} epochs), "
                     "while specified `num_epochs` is {} (total steps {})".format(start_epoch, start_step,
                                                                                  args.num_epochs,
                                                                                  total_training_steps))
        sys.exit(1)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=args.cuda_ids)

    # Initialize performance tracking and evaluate model before training
    best_value = 1e16 if args.key_metric in NEG_METRICS else -1e16  # initialize with +inf or -inf depending on key metric
    running_metrics = []  # (for validation) list of lists: for every evaluation, stores metrics like loss, MRR, MAP, ...
    best_metrics = {}

    logger.info("\n\n***** Initial evaluation on dev set *****".format(global_step))
    val_metrics, best_metrics, best_value = validate(args, model, val_dataloader, tb_writer, best_metrics, best_value, global_step, fairrmetric=fairrmetric)
    metrics_names, metrics_values = zip(*val_metrics.items())
    running_metrics.append(list(metrics_values))

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
    for i, s in enumerate(scheduler.get_last_lr()):
        logger.debug('Learning rate ({}): {}'.format(sstr[i], s))

    train_loss = 0  # this is the training loss accumulated from the beginning of training
    logging_loss = 0  # this is synchronized with `train_loss` every args.logging_steps
    model.zero_grad()
    model.train()
    # model.encoder.requires_grad_(False)  # TODO: DEBUG. REMOVE!!!!
    score_params = [(name, p) for name, p in model.named_parameters() if name.startswith('score')]
    epoch_iterator = trange(start_epoch, int(args.num_epochs), desc="Epochs")

    batch_times = utils.Timer()  # average time for the model to train (forward + backward pass) on a single batch of queries
    for epoch_idx in epoch_iterator:
        epoch_start_time = time.time()
        batch_iterator = tqdm(train_dataloader, desc="Batches")
        for step, (model_inp, _, _) in enumerate(batch_iterator):  # step can be a "sub-step", if grad. accum. > 1
            if args.resume and ((epoch_idx * epoch_steps) + step < args.grad_accum_steps * global_step):
                # TODO: one can set a state train_dataloader.dataset.skip = True, which will cause the __getitem__ to immediately return None
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

                if global_step == args.encoder_delay:  # here global_step > start_step is guaranteed
                    optimizer.add_optimizer(encoder_optimizer)
                    scheduler.add_scheduler(schedulers['encoder_scheduler'])
                    if args.reduce_on_plateau:
                        ROP_scheduler.add_scheduler(schedulers['ROP_encoder_scheduler'])
                    logger.info('Added optimizer and scheduler for query encoder')

                # logging for training
                if args.logging_steps and (global_step % args.logging_steps == 0):
                    for s in range(len(scheduler.schedulers)):
                        # first brackets select scheduler, second the group
                        tb_writer.add_scalar('learn_rate{}'.format(s), scheduler.get_last_lr()[s][0], global_step)
                    cur_loss = (train_loss - logging_loss) / args.logging_steps  # mean loss over last args.logging_steps (smoothened "current loss")
                    tb_writer.add_scalar('train/loss', cur_loss, global_step)

                    if args.debug:
                        logger.debug("Mean loss over {} steps: {:.5f}".format(args.logging_steps, cur_loss))
                        for i, s in enumerate(scheduler.get_last_lr()):
                            logger.debug('Learning rate ({}): {}'.format(sstr[i], s))
                        logger.debug("Current memory usage: {} MB or {} MB".format(np.round(utils.get_current_memory_usage()),
                                                                          np.round(utils.get_current_memory_usage2())))
                        logger.debug("Max memory usage: {} MB".format(int(np.ceil(utils.get_max_memory_usage()))))

                        logger.debug("Average lookup time: {} s /samp".format(lookup_times.get_average()))
                        logger.debug("Average retr. candidates time: {} s /samp".format(retrieve_candidates_times.get_average()))
                        logger.debug("Average prep. docids time: {} s /samp".format(prep_docids_times.get_average()))
                        logger.debug("Average sample fetching time: {} s /samp".format(sample_fetching_times.get_average()))
                        logger.debug("Average collation time: {} s /batch".format(collation_times.get_average()))
                        logger.debug("Average total batch processing time: {} s /batch".format(batch_times.get_average()))

                        logger.debug("Score parameters: {}".format(score_params))  # TODO: DEBUG

                    logging_loss = train_loss

                # evaluate at specified interval or if this is the last step
                if (args.validation_steps and (global_step % args.validation_steps == 0)) or global_step == total_training_steps:

                    logger.info("\n\n***** Running evaluation of step {} on dev set *****".format(global_step))
                    val_metrics, best_metrics, best_value = validate(args, model, val_dataloader, tb_writer,
                                                                     best_metrics, best_value, global_step,
                                                                     fairrmetric=fairrmetric)
                    metrics_names, metrics_values = zip(*val_metrics.items())
                    running_metrics.append(list(metrics_values))

                    if args.reduce_on_plateau:
                        ROP_scheduler.step(val_metrics[args.reduce_on_plateau])

                if (args.save_steps and (global_step % args.save_steps == 0)) or global_step == total_training_steps:
                    # Save model checkpoint
                    utils.save_model(os.path.join(args.save_dir, 'model_{}.pth'.format(global_step)),
                                     global_step, model, optimizer, scheduler)
                    utils.remove_oldest_checkpoint(args.save_dir, args.num_keep)

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
                " ({:.6f}s per sample)".format(train_batch_size, avg_batch_time, avg_batch_time / train_batch_size))
    logger.info("Average time to train on 1 batch ({} samples): {:.6f} sec".format(train_batch_size, batch_times.get_average()))
    logger.info('Best {} was {}. Other metrics: {}'.format(args.key_metric, best_value, best_metrics))

    logger.debug("Average lookup time: {} s".format(lookup_times.get_average()))
    logger.debug("Average retr. candidates time: {} s".format(retrieve_candidates_times.get_average()))
    logger.debug("Average prep. docids time: {} s".format(prep_docids_times.get_average()))
    logger.debug("Average sample fetching time: {} s".format(sample_fetching_times.get_average()))
    logger.debug("Average collation time: {} s".format(collation_times.get_average()))

    logger.info("Current memory usage: {} MB".format(np.round(utils.get_current_memory_usage())))
    logger.info("Max memory usage: {} MB".format(int(np.ceil(utils.get_max_memory_usage()))))

    return best_metrics


def validate(args, model, val_dataloader, tensorboard_writer, best_metrics, best_value, global_step, fairrmetric=None):
    """Run an evaluation on the validation set while logging metrics, and handle result"""

    model.eval()
    eval_start_time = time.time()
    val_metrics, ranked_df = evaluate(args, model, val_dataloader, fairrmetric=fairrmetric)
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
        # to save space: optimizer, scheduler not saved! Only latest checkpoints can be used for resuming training perfectly
        utils.save_model(os.path.join(args.save_dir, 'model_best.pth'), global_step, model)
        best_metrics = val_metrics.copy()

        if not args.no_predictions:
            ranked_filepath = os.path.join(args.pred_dir, 'best.ranked.dev.tsv')
            ranked_df.to_csv(ranked_filepath, header=False, sep='\t')

        # Export metrics to a file accumulating best records from the current experiment
        rec_filepath = os.path.join(args.pred_dir, 'training_session_records.xls')
        utils.register_record(rec_filepath, args.initial_timestamp, args.experiment_name, best_metrics)

    return val_metrics, best_metrics, best_value


def evaluate(args, model, dataloader, fairrmetric=None):
    """
    Evaluate a given model on the dataset contained in the given dataloader and compile a dataframe with
    document ranks and scores for each query. If the dataset includes relevance labels (qrels), then metrics
    such as MRR, MAP etc will be additionally computed.
    :return:
        eval_metrics: dict containing metrics (at least 1, batch processing time)
        rank_df: dataframe with indexed by qID (shared by multiple rows) and columns: PID, rank, score
    """
    qrels = dataloader.dataset.qrels  # dict{qID: dict{pID: relevance}}
    labels_exist = qrels is not None

    # num_docs is the (potentially variable) number of candidates per query
    relevances = []  # (total_num_queries) list of (num_docs) lists with non-zeros at the indices corresponding to actually relevant passages
    num_relevant = []  # (total_num_queries) list of number of ground truth relevant documents per query
    df_chunks = []  # (total_num_queries) list of dataframes, each with index a single qID and corresponding (num_docs) columns PID, rank, score
    query_time = 0  # average time for the model to score candidates for a single query
    total_loss = 0  # total loss over dataset

    with torch.no_grad():
        for batch_data, qids, docids in tqdm(dataloader, desc="Evaluating"):
            batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
            start_time = time.perf_counter()
            out = model(**batch_data)
            query_time += time.perf_counter() - start_time
            rel_scores = out['rel_scores'].detach().cpu().numpy()  # (batch_size, num_docs) relevance scores in [0, 1]
            if 'loss' in out:
                total_loss += out['loss'].sum().item()
            assert len(qids) == len(docids) == len(rel_scores)

            # Rank documents based on their scores
            num_docs_per_query = [len(cands) for cands in docids]
            num_lengths = set(num_docs_per_query)
            no_padding = (len(num_lengths) == 1)  # whether all queries in this batch had the same number of candidates

            if no_padding:  # (only) 10% speedup compared to other case
                docids_array = np.array(docids, dtype=np.int32)  # (batch_size, num_docs) array of docIDs per query
                # First shuffle along doc dimension, because relevant document(s) are placed at the beginning and would benefit
                # in case of score ties! (can happen e.g. with saturating score functions)
                inds = np.random.permutation(rel_scores.shape[1])
                np.take(rel_scores, inds, axis=1, out=rel_scores)
                np.take(docids_array, inds, axis=1, out=docids_array)

                # Sort by descending relevance
                inds = np.fliplr(np.argsort(rel_scores, axis=1))  # (batch_size, num_docs) inds to sort rel_scores
                # (batch_size, num_docs) docIDs per query, in order of descending relevance score
                ranksorted_docs = np.take_along_axis(docids_array, inds, axis=1)
                sorted_scores = np.take_along_axis(rel_scores, inds, axis=1)
            else:
                # (batch_size) iterables of docIDs and scores per query, in order of descending relevance score
                ranksorted_docs, sorted_scores = zip(*(map(rank_docs, docids, rel_scores)))

            # extend by batch_size elements
            df_chunks.extend(pd.DataFrame(data={"PID": ranksorted_docs[i],
                                                "rank": list(range(1, len(docids[i]) + 1)),
                                                "score": sorted_scores[i]},
                                          index=[qids[i]] * len(docids[i])) for i in range(len(qids)))

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

    ## evaluate fairness  # TODO: split into separate function
    if fairrmetric is not None:
        try:
            _retrievalresults = {}
            for _item in df_chunks:
                _qid = _item.index[0]
                _retrievalresults[_qid] = _item['PID'].tolist()

            eval_FaiRR, eval_NFaiRR = fairrmetric.calc_FaiRR_retrievalresults(retrievalresults=_retrievalresults)

            # for _cutoff in eval_FaiRR:
            #    eval_metrics['FaiRR_cutoff_%d' % _cutoff] = eval_FaiRR[_cutoff]
            for _cutoff in eval_FaiRR:
                eval_metrics['NFaiRR_cutoff_%d' % _cutoff] = eval_NFaiRR[_cutoff]
        except:
            logger.error('Fairness metrics calculation failed!')

    if labels_exist and (args.debug or args.task != 'train'):
        try:
            rs = (np.nonzero(r)[0] for r in relevances)
            ranks = [1 + int(r[0]) if r.size else 1e10 for r in rs]  # for each query, what was the rank of the rel. doc
            freqs, bin_edges = np.histogram(ranks, bins=[1, 5, 10, 20, 30] + list(range(50, 1050, 50)))
            bin_labels = ["[{}, {})".format(bin_edges[i], bin_edges[i + 1])
                          for i in range(len(bin_edges) - 1)] + ["[{}, inf)".format(bin_edges[-1])]
            logger.info('\nHistogram of ranks for the ground truth documents:\n')
            utils.ascii_bar_plot(bin_labels, freqs, width=50, logger=logger)
        except:
            logger.error('Not possible!')

    return eval_metrics, ranked_df


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
    """Can handle multiple levels of relevance.
    Args:
        gt_relevant: for a given query, it's a dict mapping from ground-truth relevant passage ID to level of relevance
        candidates: list of candidate pids
        max_docs: consider only the first this many documents
    Returns: list of length min(max_docs, len(pred)) with non-zero at the indices corresponding to passages in `gt_relevant`
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


# Very inefficient. Used by RepBERT only.
def evaluate_slow(args, model, dataloader, mode, prefix='model'):
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
                scores = torch.diagonal(outputs['rel_scores']).detach().cpu().numpy()  # outputs[0]
                assert len(qids) == len(docids) == len(scores)
                for qid, docid, score in zip(qids, docids, scores):
                    outputfile.write(f"{qid}\t{docid}\t{score}\n")

    rank_output = os.path.join(args.pred_dir, prefix + ".{}.rank.tsv".format(mode))
    utils.generate_rank(output_file_path, rank_output)

    if mode == "dev":
        mrr = utils.eval_results(rank_output)
        return mrr


def inspect(args, model, dataloader):
    """
    Interactively examine an existing ranking of candidates by the specified model, alongside with the respective
    original queries and documents, reconstructed tokenizations, embeddings, ground truth relevant documents,
    and a (raw .tsv) reference ranked candidates file, which may or may not correspond to the same ranking
    as `eval_candidates_path`
    """

    # to have agreement between loaded and computed embeddings, sequence lengths must correspond to the max. length used for precomputed embeddings
    QUERY_LENGTH = args.max_query_length
    DOC_LENGTH = args.max_doc_length
    DISPLAY_DOCS = 10  # how many candidate documents (from the top and bottom) to display and use for integrity calculation
    ERROR_TOLERANCE = 1e-4  # torerable absolute difference in embeddings
    np.set_printoptions(linewidth=200)
    torch.set_printoptions(linewidth=200)
    pd.options.display.width = 400  # will automatically detect console width
    pd.set_option('display.max_colwidth', 0)

    # Modify existing console handler
    logging.getLogger().removeHandler(logging.getLogger().handlers[0])
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    qrels = dataloader.dataset.qrels  # dict{qID: dict{pID: relevance}}
    labels_exist = qrels is not None

    # num_docs is the (potentially variable) number of candidates per query
    relevances = []  # (total_num_queries) list of (num_docs) lists with non-zeros at the indices corresponding to actually relevant passages
    num_relevant = []  # (total_num_queries) list of number of ground truth relevant documents per query
    df_chunks = []  # (total_num_queries) list of dataframes, each with index a single qID and corresponding (num_docs) columns PID, rank, score
    query_time = 0  # average time for the model to score candidates for a single query
    total_loss = 0  # total loss over dataset
    error_queries = []  # query IDs corresponding to tokenization or embedding errors
    error_docs = set([])  # document IDs corresponding to tokenization or embedding errors
    num_inspected_docs = 0  # total number of inspected documents
    total_num_score_errors = 0  # total number of errors in scores
    total_queries = 0  # total number of inspected queries

    logger.info("Loading raw queries ...")
    query_dict = load_original_sequences(args.raw_queries_path)
    if args.query_emb_memmap_dir:
        logger.info("Loading query embeddings from '{}' ...".format(args.query_emb_memmap_dir))
        query_embeddings = EmbeddedSequences(embedding_memmap_dir=args.query_emb_memmap_dir, seq_type='query')

    logger.info("Loading raw documents ...")
    doc_dict = load_original_sequences(args.raw_collection_path)

    if args.collection_memmap_dir:
        logger.info("Loading tokenized documents collection from '{}' ...".format(args.collection_memmap_dir))
        doc_tokens_collection = MSMARCO_DocDataset(args.collection_memmap_dir, DOC_LENGTH, dataloader.dataset.tokenizer)

    aggregation_func = get_aggregation_function(args.query_aggregation)

    with torch.no_grad():
        for batch_data, qids, docids in tqdm(dataloader, desc="Evaluating"):
            batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
            start_time = time.perf_counter()
            out = model(**batch_data)
            query_time += time.perf_counter() - start_time
            rel_scores = out['rel_scores'].detach().cpu().numpy()  # (batch_size, num_docs) relevance scores in [0, 1]
            if 'loss' in out:
                total_loss += out['loss'].sum().item()
            assert len(qids) == len(docids) == len(rel_scores)

            # Calculate and extract encoder final query representations
            encoder_out = model.encoder(batch_data['query_token_ids'].to(torch.int64),
                                        attention_mask=batch_data['query_token_ids'])
            enc_hidden_states = encoder_out['last_hidden_state']  # (batch_size, max_query_len, query_dim)

            # Rank documents based on their scores
            num_docs_per_query = [len(cands) for cands in docids]
            num_lengths = set(num_docs_per_query)
            no_padding = (len(num_lengths) == 1)  # whether all queries in this batch had the same number of candidates

            if no_padding:  # (only) 10% speedup compared to other case
                docids_array = np.array(docids, dtype=np.int32)  # (batch_size, num_docs) array of docIDs per query
                # First shuffle along doc dimension, because relevant document(s) are placed at the beginning and would benefit
                # in case of score ties! (can happen e.g. with saturating score functions)
                inds = np.random.permutation(rel_scores.shape[1])
                np.take(rel_scores, inds, axis=1, out=rel_scores)
                np.take(docids_array, inds, axis=1, out=docids_array)

                # Sort by descending relevance
                inds = np.fliplr(np.argsort(rel_scores, axis=1))  # (batch_size, num_docs) inds to sort rel_scores
                # (batch_size, num_docs) docIDs per query, in order of descending relevance score
                ranksorted_docs = np.take_along_axis(docids_array, inds, axis=1)
                sorted_scores = np.take_along_axis(rel_scores, inds, axis=1)
            else:
                # (batch_size) iterables of docIDs and scores per query, in order of descending relevance score
                ranksorted_docs, sorted_scores = zip(*(map(rank_docs, docids, rel_scores)))

            # extend by batch_size elements
            batch_dfs = [pd.DataFrame(data={"PID": ranksorted_docs[i],
                                            "rank": list(range(1, len(docids[i]) + 1)),
                                            "score": sorted_scores[i]},
                                      index=[qids[i]] * len(docids[i])) for i in range(len(qids))]
            df_chunks.extend(batch_dfs)

            if labels_exist:
                relevances.extend(get_relevances(qrels[qids[i]], ranksorted_docs[i]) for i in range(len(qids)))
                num_relevant.extend(len(qrels[qid]) for qid in qids)

            # Display input and output information for retrieval on each query
            for i in range(len(qids)):  # iterate over batch_size
                total_queries += 1
                # Check integrity of queries
                query_ok = True
                logger.info("Query ID: {}".format(qids[i]))
                logger.info("Original Text: {}".format(query_dict[qids[i]]))

                logger.debug("Input query token IDs (as in '{}'):\n{}".format(args.tokenized_path,
                                                                              batch_data['query_token_ids'][i, :]))
                logger.debug("Rec. input query tokens:\n{}".format(
                    dataloader.dataset.tokenizer.convert_ids_to_tokens(batch_data['query_token_ids'][i, :])))
                start_time = time.perf_counter()
                tokenized_query = dataloader.dataset.tokenizer([query_dict[qids[i]]], max_length=QUERY_LENGTH, padding=True, return_tensors="pt")
                tokenization_time = time.perf_counter() - start_time
                logger.debug("Time to tokenize: {} s".format(tokenization_time))
                tokenized_query = {k: v.to(args.device) for k, v in tokenized_query.items()}
                qtokenized_ids = tokenized_query["input_ids"][0]
                logger.debug("Tokenized query IDs: {}".format(qtokenized_ids))
                qtokenized_tokens = dataloader.dataset.tokenizer.convert_ids_to_tokens(qtokenized_ids)
                logger.debug("Query tokens: {}".format(qtokenized_tokens))

                if not (torch.equal(qtokenized_ids, batch_data['query_token_ids'][i, :len(qtokenized_ids)])):
                    logger.warning("Query tokenization error!")
                    query_ok = False

                query_emb_model = aggregation_func(enc_hidden_states)[i, :]  # .detach().cpu().numpy()
                logger.debug("Query embedding (inside model):\n{}".format(query_emb_model))
                encoder_out2 = model.encoder(**tokenized_query)  # using currently tokenized query
                query_emb_model2 = aggregation_func(encoder_out2[0])  # [0] selects 'encoder_hidden_states'
                L2_diff = torch.linalg.norm(query_emb_model - query_emb_model2).detach().cpu().numpy()
                if L2_diff > ERROR_TOLERANCE:
                    logger.warning("Query embedding as calculated from pre-tokenized query inside model differs from "
                                   "the embedding as calculated directly by encoder! L2-norm of difference: {}".format(L2_diff))
                    logger.debug("Query embedding (directly from encoder):\n{}".format(query_emb_model2))
                    query_ok = False

                if args.query_emb_memmap_dir:
                    loaded_query_emb = torch.Tensor(query_embeddings[[qids[i]]]).to(args.device)
                    logger.debug(
                        "Loaded query embedding (from '{}'):\n{}".format(args.query_emb_memmap_dir, loaded_query_emb))
                    L2_diff2 = torch.linalg.norm(query_emb_model2 - loaded_query_emb).detach().cpu().numpy()
                    if L2_diff2 > ERROR_TOLERANCE:
                        logger.warning("Query embedding as calculated directly by encoder differs from "
                                       "the embedding as loaded from '{}'! L2-norm of difference: {}".format(
                            args.query_emb_memmap_dir, L2_diff2))
                        query_ok = False

                if not query_ok:
                    error_queries.append(qids[i])

                # Check integrity of documents and scores
                # Checks DISPLAY_DOCS top and bottom candidate docs, and ground-truth relevant docs, if they exist
                docs_df = batch_dfs[i]  # df with PIDs, ranks, scores for a single qID
                ranksorted_docids = list(ranksorted_docs[i])
                docs_text = [doc_dict[did] for did in ranksorted_docids]  # list of candidate doc texts (for 1 query)
                docs_df['text'] = docs_text
                docs_df['text'] = docs_df['text'].str.wrap(200)  # to set max line width for text (wrap)
                docs_to_check_text = []  # list of candidate doc texts to be checked
                docs_to_check_ids = []  # list of candidate doc IDs to be checked

                if labels_exist:
                    gt_docids = list(qrels[qids[i]].keys())
                    gt_doc_text = []
                    logger.info("Ground truth relevant document(s): ")
                    for d, did in enumerate(gt_docids):
                        print("{}: DocID:{:8}: {}\n".format(d, did, doc_dict[did]))
                        gt_doc_text.append(doc_dict[did])
                    docs_to_check_text = gt_doc_text
                    docs_to_check_ids = gt_docids

                logger.info("Top-Scored candidates:\n{}\n".format(docs_df.iloc[:DISPLAY_DOCS]))
                logger.info("Bottom-Scored candidates:\n{}\n".format(docs_df.iloc[-DISPLAY_DOCS:]))

                docs_to_check_text += docs_text[:DISPLAY_DOCS] + docs_text[-DISPLAY_DOCS:]
                docs_to_check_ids += ranksorted_docids[:DISPLAY_DOCS] + ranksorted_docids[-DISPLAY_DOCS:]
                num_inspected_docs += len(docs_to_check_ids)
                start_time = time.perf_counter()
                tokenized_docs = dataloader.dataset.tokenizer(docs_to_check_text, max_length=DOC_LENGTH, padding=True, return_tensors="pt")
                tokenization_time = time.perf_counter() - start_time
                logger.debug("Time to tokenize {} documents: {} s".format(len(docs_to_check_text), tokenization_time))

                for j in range(tokenized_docs["input_ids"].shape[0]):
                    logger.debug("DocID {}:\n".format(docs_to_check_ids[j]))
                    doc_tokens = dataloader.dataset.tokenizer.convert_ids_to_tokens(tokenized_docs["input_ids"][j])
                    logger.debug("Tokenized: {}\n".format(doc_tokens))
                    if args.collection_memmap_dir:
                        loaded_token_ids = torch.tensor(doc_tokens_collection[docs_to_check_ids[j]]['input_ids'])
                        loaded_tokens = dataloader.dataset.tokenizer.convert_ids_to_tokens(loaded_token_ids)
                        logger.debug("Loaded: {}\n\n".format(loaded_tokens))
                        if not (torch.equal(tokenized_docs["input_ids"][j][:len(loaded_token_ids)], loaded_token_ids)):
                            logger.warning("Doc tokenization error!")
                            error_docs.add(docs_to_check_ids[j])

                tokenized_docs = {k: v.to(args.device) for k, v in tokenized_docs.items()}
                model_out = model.encoder(**tokenized_docs)
                doc_emb_model = aggregation_func(model_out[0])  # (num_check_docs, d_emb) . [0] selects 'encoder_hidden_states'
                loaded_doc_emb = torch.Tensor(dataloader.dataset.emb_collection[docs_to_check_ids]).to(args.device)  # (num_check_docs, d_emb)
                abs_diffs = torch.abs(doc_emb_model - loaded_doc_emb)
                max_abs_diff = torch.max(abs_diffs)

                if max_abs_diff > ERROR_TOLERANCE:
                    logger.warning("Doc embeddings as computed directly by model differ from "
                                   "the embeddings as loaded from '{}'! Max. abs. difference: {}".format(args.embedding_memmap_dir, max_abs_diff))
                    error_inds = abs_diffs > ERROR_TOLERANCE  # (num_docs, emb_dim)
                    logger.warning("{} / {} elements have an absolute "
                                   "difference larger than {}".format(torch.sum(error_inds), torch.numel(doc_emb_model), ERROR_TOLERANCE))
                    error_inds = torch.any(error_inds, dim=1)  # (num_docs,) boolean array showing which docs differed
                    error_docs.update(docs_to_check_ids[k] for k in range(len(error_inds)) if error_inds[k])
                    logger.debug("Loaded doc embeddings (from '{}'):\n{}".format(args.embedding_memmap_dir, loaded_doc_emb))
                    logger.debug("Computed doc embeddings:\n{}".format(doc_emb_model))

                # Compare scores between MDST model output and computed encoder scores (emb. dot product)
                docs_to_check_inds = list(np.nonzero(relevances[i])[0]) if labels_exist else []  # insert indices of gt documents
                docs_to_check_inds += list(range(DISPLAY_DOCS)) + list(range(-DISPLAY_DOCS, 0, 1))
                model_scores = sorted_scores[i][docs_to_check_inds]  # scores as calculated in MDST model
                encoder_scores = torch.matmul(query_emb_model2, doc_emb_model.T).detach().cpu().numpy()  # encoder scores (emb. dot product)
                # encoder_scores = torch.matmul(query_emb_model2, loaded_doc_emb.T).detach().cpu().numpy()  # encoder scores (emb. dot product)  # TODO: just for DEBUG, to confirm that scoring module does simple dot product
                abs_diffs = np.abs(model_scores - encoder_scores)
                max_abs_diff = np.max(abs_diffs)

                if max_abs_diff > ERROR_TOLERANCE:
                    logger.warning("Scores as computed directly by encoder model (dot product) differ from "
                                   "the scores in the output of MDST model. Max. abs. difference: {}".format(max_abs_diff))
                    error_inds = abs_diffs > ERROR_TOLERANCE
                    num_score_errors = np.sum(error_inds)
                    total_num_score_errors += num_score_errors
                    logger.warning("{} / {} scores have an absolute difference larger than {}".format(num_score_errors, len(model_scores), ERROR_TOLERANCE))
                    logger.debug("Scores in the output of MDST model:\n{}\n".format(model_scores))
                    logger.debug("Scores computed by encoder model (query-doc emb. dot product):\n{}\n".format(encoder_scores))

                print()
                inp = input("Press any key to continue, or 'q' to exit: ")
                if inp == 'q':
                    break
            if inp == 'q':
                break

    if len(error_queries):
        logger.error("Errors found in {} "
                     "out of {} ({:.3f}%) queries:\n{}".format(len(error_queries), total_queries,
                                                               100*len(error_queries)/total_queries,
                                                               error_queries))
        for q in error_queries:
            logger.debug("QID:{:7}: {}".format(q, query_dict[q]))
    else:
        logger.info("All {} queries ok!".format(len(dataloader.dataset.qids)))
    print()

    if len(error_docs):
        logger.error("Errors found in {} "
                     "out of {} ({:.3f}%) inspected documents:\n{}".format(len(error_docs), num_inspected_docs,
                                                                           100*len(error_docs)/num_inspected_docs,
                                                                           error_docs))
        for d in error_docs:
            logger.debug("docID:{:8}: {}".format(d, doc_dict[d]))
    else:
        logger.info("All {} documents ok!".format(num_inspected_docs))
    print()

    if total_num_score_errors:
        logger.error("Errors found in {} score calculations".format(total_num_score_errors))
    else:
        logger.info("All scores are consistent!")

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

    if labels_exist and (args.debug or args.task != 'train'):
        try:
            rs = (np.nonzero(r)[0] for r in relevances)
            ranks = [1 + int(r[0]) if r.size else 1e10 for r in rs]  # for each query, what was the rank of the rel. doc
            freqs, bin_edges = np.histogram(ranks, bins=[1, 5, 10, 20, 30] + list(range(50, 1050, 50)))
            bin_labels = ["[{}, {})".format(bin_edges[i], bin_edges[i + 1])
                          for i in range(len(bin_edges) - 1)] + ["[{}, inf)".format(bin_edges[-1])]
            logger.info('\nHistogram of ranks for the ground truth documents:\n')
            utils.ascii_bar_plot(bin_labels, freqs, width=50, logger=logger)
        except:
            logger.error('Not possible!')

    return eval_metrics, ranked_df


def main(config):
    args = utils.dict2obj(config)  # Convert config dict to args object

    if args.debug:
        logger.setLevel('DEBUG')
    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'output.log'))
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run

    # Setup CUDA, GPU
    args.n_gpu = len(args.cuda_ids)
    if torch.cuda.is_available():
        if args.n_gpu > 1:
            args.device = torch.device("cuda")
        else:
            args.device = torch.device("cuda:%d" % args.cuda_ids[0])
    else:
        args.device = torch.device("cpu")

    # Log current hardware setup
    logger.info("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    if args.device.type == 'cuda':
        logger.info("Device: {}".format(torch.cuda.get_device_name(0)))
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        logger.info("Total memory: {} MB".format(np.ceil(total_mem)))


    # ##### TODO: HACK TO DEBUG hyperparam_optim
    #
    # def fitness_func(args):
    #     fitness = - 1e-3*np.abs(args.warmup_steps - 5000) - 1e5*args.weight_decay + 1e-4*args.num_candidates*args.num_random_neg \
    #               - 2e-3*(args.num_candidates - 200)**2 + 1e-2*max(512, args.d_model) - (args.dim_feedforward - 1.5*args.d_model)**2
    #     return fitness
    #
    # time.sleep(0.1)
    # fitness = fitness_func(args)
    # output = {'Recall': 2*fitness, 'MRR': fitness}
    # logger.info("Done! fitness: {}".format(fitness))
    # return output
    # ##### TODO: HACK TO DEBUG hyperparam_optim

    # Set seed
    utils.set_seed(args)

    # Get tokenizer
    tokenizer = get_tokenizer(args)

    # Load evaluation set and initialize evaluation dataloader
    if args.task == 'train' or args.task == 'inspect':
        eval_mode = 'dev'  # 'eval' here is the name of the MSMARCO test set, 'dev' is the validation set
    else:
        eval_mode = args.task

    logger.info("Preparing {} dataset ...".format(eval_mode))
    start_time = time.time()
    eval_dataset = get_dataset(args, eval_mode, tokenizer)  # CHANGED here from eval_mode
    collate_fn = eval_dataset.get_collate_func(n_gpu=args.n_gpu)
    logger.info("'{}' data loaded in {:.3f} sec".format(eval_mode, time.time() - start_time))

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                 num_workers=args.data_num_workers, collate_fn=collate_fn)

    logger.info("Number of {} samples: {}".format(eval_mode, len(eval_dataset)))
    logger.info("Batch size: %d", args.eval_batch_size)

    # initialize fairness
    fairrmetric = None
    if args.collection_neutrality_path is not None:
        logger.info("Loading FaiRRMetric ...")
        _fairrmetrichelper = FaiRRMetricHelper()
        _background_doc_set = _fairrmetrichelper.read_documentset_from_retrievalresults(args.background_set_runfile_path)
        fairrmetric = FaiRRMetric(args.collection_neutrality_path, _background_doc_set)

    # Initialize model. This is done after loading the data, to know the doc. embeddings dimension
    logger.info("Initializing model ...")
    if args.model_type == 'repbert':
        # keep configuration setup like RepBERT (for backward compatibility).
        # The model is a common/shared BERT query-document encoder, without interactions between query and document token representations
        if args.load_model_path is None:
            args.load_model_path = "bert-base-uncased"
        # Works with either directory path containing HF config file, or JSON HF config file,  or pre-defined model string
        config_obj = BertConfig.from_pretrained(args.load_model_path)
        model = RepBERT_Train.from_pretrained(args.load_model_path, config=config_obj)
    else:  # new configuration setup for MultiDocumentScoringTransformer models
        model = get_model(args, eval_dataset.emb_collection.embedding_vectors.shape[1])

    logger.debug("Model:\n{}".format(model))
    logger.info("Total number of model parameters: {}".format(utils.count_parameters(model)))
    logger.info("Total trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))
    logger.info("Number of encoder parameters: {}".format(utils.count_parameters(model.encoder)))
    logger.info("Trainable encoder parameters: {}".format(utils.count_parameters(model.encoder, trainable=True)))

    model.to(args.device)  # will also print model architecture, besides moving to GPU

    if args.task == "train":
        return train(args, model, eval_dataloader, tokenizer, fairrmetric=fairrmetric)
    else:
        # Just evaluate trained model on some dataset (needs ~27GB for MS MARCO dev set)

        # only composite (non-repbert) models need to be loaded; repbert is already loaded at this point
        if args.load_model_path and (args.model_type != 'repbert'):
            model, global_step, _, _ = utils.load_model(model, args.load_model_path, device=args.device)

        logger.info("Will evaluate model on candidates in: {}".format(args.eval_candidates_path))

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=args.cuda_ids)

        model.eval()
        # Run evaluation
        eval_start_time = time.time()
        if args.task == 'inspect':
            # Interactive inspection mode
            logger.info("Interactive inspection mode")
            eval_metrics, ranked_df = inspect(args, model, eval_dataloader)
        else:
            eval_metrics, ranked_df = evaluate(args, model, eval_dataloader, fairrmetric=fairrmetric)
        eval_runtime = time.time() - eval_start_time
        logger.info("Evaluation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))
        print()
        print_str = 'Evaluation Summary: '
        for k, v in eval_metrics.items():
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)

        if args.inject_ground_truth:
            logger.warning("Ground truth documents were injected among candidates! This may cause inflated metrics!")

        filename = 'reranked_' + os.path.basename(args.eval_candidates_path)
        if not filename.endswith('.tsv'):
            filename = filename[:filename.find('_memmap')] + '.tsv'  # it "eats" the last character if not '_memmap'
        ranked_filepath = os.path.join(args.pred_dir, filename)
        logger.info("Writing predicted ranking to: {} ...".format(ranked_filepath))
        ranked_df.to_csv(ranked_filepath, header=False, sep='\t')

        # Export record metrics to a file accumulating records from all experiments
        utils.register_record(args.records_file, args.initial_timestamp, args.experiment_name,
                              eval_metrics, comment=args.comment)

        return eval_metrics


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    config = utils.load_config(args)  # configuration dictionary
    config = options.check_args(config)  # check validity of settings and make necessary conversions

    # Create output directory and subdirectories
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(output_dir))

    output_dir = os.path.join(output_dir, config['experiment_name'])

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    if (not config['no_timestamp']) or (len(config['experiment_name']) == 0):
        rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
        output_dir += "_" + formatted_timestamp + "_" + rand_suffix
    config['output_dir'] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)
    logger.info("Stored configuration file in '{}'".format(output_dir))

    # Create subdirectories and store additional configuration info
    info_dict = {'initial_timestamp': formatted_timestamp,
                 'save_dir': os.path.join(output_dir, 'checkpoints'),
                 'pred_dir': os.path.join(output_dir, 'predictions'),
                 'tensorboard_dir': os.path.join(output_dir, 'tb_summaries')}
    utils.create_dirs([info_dict['save_dir'], info_dict['pred_dir'], info_dict['tensorboard_dir']])
    with open(os.path.join(output_dir, 'info.txt'), 'w') as fp:
        json.dump(info_dict, fp)
    config.update(info_dict)

    return config


def get_dataset(args, eval_mode, tokenizer):
    """Initialize and return dataset object based on args"""

    if args.model_type == 'repbert':
        return MSMARCODataset(eval_mode, args.msmarco_dir, args.collection_memmap_dir, args.tokenized_path,
                              args.max_query_length, args.max_doc_length, limit_size=args.eval_limit_size)
    else:
        return MYMARCO_Dataset(eval_mode, args.embedding_memmap_dir, args.tokenized_path, args.eval_candidates_path,
                               qrels_path=args.qrels_path, tokenizer=tokenizer,
                               max_query_length=args.max_query_length, num_candidates=args.num_candidates,
                               limit_size=args.eval_limit_size,
                               load_collection_to_memory=args.load_collection_to_memory,
                               inject_ground_truth=args.inject_ground_truth,
                               query_ids_path=args.eval_query_ids)


def get_query_encoder(query_encoder_from, query_encoder_config):
    """Initialize and return query encoder model object based on args"""

    if os.path.exists(query_encoder_from):
        logger.info("Will load pre-trained query encoder from: {}".format(query_encoder_from))
    else:
        logger.warning("Will initialize standard HuggingFace '{}' as a query encoder!".format(query_encoder_from))
    start_time = time.time()
    encoder = AutoModel.from_pretrained(query_encoder_from, config=query_encoder_config)
    logger.info("Query encoder loaded in {} s".format(time.time() - start_time))
    return encoder


def get_model(args, doc_emb_dim=None):
    """Initialize and return end-to-end model object based on args"""

    query_encoder = get_query_encoder(args.query_encoder_from, args.query_encoder_config)

    if args.model_type == 'mdstransformer':

        loss_module = get_loss_module(args.loss_type, args)
        aux_loss_module = None
        if args.aux_loss_type is not None:
            aux_loss_module = get_loss_module(args.aux_loss_type, args)  # instantiate auxiliary loss module

        return MDSTransformer(custom_encoder=query_encoder,
                              d_model=args.d_model,
                              num_heads=args.num_heads,
                              num_decoder_layers=args.num_layers,
                              dim_feedforward=args.dim_feedforward,
                              dropout=args.dropout,
                              activation=args.activation,
                              normalization=args.normalization_layer,
                              doc_emb_dim=doc_emb_dim,
                              scoring_mode=args.scoring_mode,
                              query_emb_aggregation=args.query_aggregation,
                              loss_module=loss_module,
                              aux_loss_module=aux_loss_module,
                              aux_loss_coeff=args.aux_loss_coeff,
                              selfatten_mode=args.selfatten_mode,
                              no_decoder=args.no_decoder,
                              no_dec_crossatten=args.no_dec_crossatten,
                              bias_regul_coeff=args.bias_regul_coeff,
                              bias_regul_cutoff=args.bias_regul_cutoff)
    else:
        raise NotImplementedError('Unknown model type')


def get_tokenizer(args):
    """Initialize and return tokenizer object based on args"""

    if args.tokenizer_from is None:  # use same config as specified for the query encoder model
        return AutoTokenizer.from_pretrained(args.query_encoder_from, config=args.query_encoder_config)
    else:
        return AutoTokenizer.from_pretrained(args.tokenizer_from)


if __name__ == "__main__":
    total_start_time = time.time()
    args = run_parse_args()
    config = setup(args)  # Setup experiment session
    main(config)
    logger.info("All done!")
    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))
