import logging
logging.basicConfig(format='%(asctime)s | %(name)-8s - %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Loading packages ...")
import os
import sys
import random
import time
import json
from datetime import datetime
import string
from collections import OrderedDict
import bisect
import ipdb

import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler
# from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers import BertConfig, AutoTokenizer, AutoModel
import optuna

# Package modules
import options
from options import *
from modeling import RepBERT_Train, CODER, get_loss_module
from dataset import MYMARCO_Dataset
from dataset import lookup_times, sample_fetching_times, collation_times, retrieve_candidates_times, prep_docids_times
from optimizers import get_optimizers, MultiOptimizer, get_schedulers, MultiScheduler
import utils
from utils import calculate_metrics, get_relevances, rank_docs, get_retrieval_metrics
from inspect_pipeline import inspect
from fair_retrieval.metrics_FaiRR import FaiRRMetric, FaiRRMetricHelper

val_times = utils.Timer()  # stores measured validation times

STEP_THRESHOLD = 0  #4000 # Used for fairness regularization; checkpoints corresponding to best performance before STEP_THRESHOLD steps will be ignored


def train(args, model, val_dataloader, tokenizer=None, fairrmetric=None, trial=None):
    """
    Prepare training dataset, train the model and handle results.
    fairrmetric is an optional object for evaluating fairness/neutrality of ranked documents.
    trial is an optional Optuna hyperparameter optimization object
    """

    tb_writer = SummaryWriter(args.tensorboard_dir)

    train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    logger.info("Preparing {} dataset ...".format('train'))
    start_time = time.time()
    train_dataset = MYMARCO_Dataset('train', args.embedding_memmap_dir, args.tokenized_path, args.train_candidates_path,
                                    qrels_path=args.qrels_path, target_scores_path=args.target_scores_path, query_ids_path=args.train_query_ids,
                                    tokenizer=tokenizer, max_query_length=args.max_query_length,
                                    num_candidates=args.num_candidates, limit_size=args.train_limit_size,
                                    load_collection_to_memory=args.load_collection_to_memory,
                                    emb_collection=val_dataloader.dataset.emb_collection,
                                    relevance_labels_mapping=args.relevance_labels_mapping,
                                    include_at_level=args.include_at_level, relevant_at_level=args.relevant_at_level,
                                    max_inj_relevant=args.max_inj_relevant,
                                    label_normalization=args.label_normalization, boost_relevant=args.boost_relevant, boost_factor=args.boost_factor,
                                    collection_neutrality_path=args.collection_neutrality_path)
    collate_fn = train_dataset.get_collate_func(num_random_neg=args.num_random_neg, n_gpu=args.n_gpu,
                                                label_format=model.loss_module.formatting, label_normalization=None)
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

    logger.info("Current memory usage: {} MB".format(int(np.round(utils.get_current_memory_usage()))))
    logger.info("Max memory usage: {} MB".format(int(np.ceil(utils.get_max_memory_usage()))))

    # Initialize performance tracking and evaluate model before training
    best_values = []  # sorted list of length args.num_keep_best, containing the top args.num_keep_best performance metrics
    best_steps = []  # list containing the global step number corresponding to best_values
    running_metrics = []  # (for validation) list of lists: for every evaluation, stores metrics like loss, MRR, MAP, ...

    logger.info("\n\n***** Initial evaluation on dev set ***** (step {})".format(global_step))
    val_metrics, best_values, best_steps, predictions = validate(args, model, val_dataloader, tb_writer, best_values, best_steps, global_step, fairrmetric=fairrmetric)
    best_metrics = val_metrics.copy()  # dict of all monitored metrics at the step with the best args.key_metric
    metrics_names, metrics_values = zip(*val_metrics.items())
    running_metrics.append(list(metrics_values))
    ranked_filepath = os.path.join(args.pred_dir, 'original_top1000.dev.' + args.write_predictions)
    utils.write_predictions(ranked_filepath, predictions, format=args.write_predictions)
    freqs_orig, bins_orig = utils.plot_rank_barplot(val_dataloader.dataset.reformatted_qrels, predictions,
                                                    relevance_thr=args.eval_relevant_at_level,
                                                    save_as=os.path.join(args.output_dir, "orig_rank_historgram.pdf"))

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
    # score_params = [(name, p) for name, p in model.named_parameters() if name.startswith('score')]
    epoch_iterator = trange(start_epoch, int(args.num_epochs), desc="Epochs")

    should_stop = False  # flag to stop training
    batch_times = utils.Timer()  # average time for the model to train (forward + backward pass) on a single batch of queries
    for epoch_idx in epoch_iterator:
        epoch_start_time = time.time()
        batch_iterator = tqdm(train_dataloader, desc="Batches")
        for step, (model_inp, _, _) in enumerate(batch_iterator):  # step can be a "sub-step", if grad. accum. > 1
            if should_stop:
                break
            if args.resume and ((epoch_idx * epoch_steps) + step < args.grad_accum_steps * global_step):
                # TODO: one can set a state train_dataloader.dataset.skip = True, which will cause the __getitem__ to immediately return None
                continue  # this is done to continue dataloader from the correct step, when using args.resume

            model_inp = {k: v.to(args.device) for k, v in model_inp.items()}
            start_time = time.perf_counter()
            try:
                output = model(**model_inp)  # model output is a dictionary
            except RuntimeError:
                raise optuna.exceptions.TrialPruned()
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
                    avg_batch_time = batch_times.get_average()
                    tb_writer.add_scalar('train/batch time', avg_batch_time, global_step)
                    tb_writer.add_scalar('train/epoch', epoch_idx, global_step)
                    if model.score_cands.temperature is not None:
                        tb_writer.add_scalar('train/temperature', model.score_cands.temperature, global_step)
                        tb_writer.add_scalar('train/score_bias', model.score_cands.b, global_step)

                    logger.info("Mean loss over {} steps: {:.5f}".format(args.logging_steps, cur_loss))
                    if args.debug:
                        for i, s in enumerate(scheduler.get_last_lr()):
                            logger.debug('Learning rate ({}): {}'.format(sstr[i], s))
                        logger.debug("Current memory usage: {} MB".format(int(np.round(utils.get_current_memory_usage()))))
                        logger.debug("Max memory usage: {} MB".format(int(np.ceil(utils.get_max_memory_usage()))))

                        logger.debug("Average lookup time: {} s /samp".format(lookup_times.get_average()))
                        logger.debug("Average retr. candidates time: {} s /samp".format(retrieve_candidates_times.get_average()))
                        logger.debug("Average prep. docids time: {} s /samp".format(prep_docids_times.get_average()))
                        logger.debug("Average sample fetching time: {} s /samp".format(sample_fetching_times.get_average()))
                        logger.debug("Average collation time: {} s /batch".format(collation_times.get_average()))
                        # logger.debug("Score parameters: {}".format(score_params))
                    logger.info("Average total batch processing time: {} s /batch".format(avg_batch_time))

                    logging_loss = train_loss

                # evaluate at specified interval or if this is the last step
                if (args.validation_steps and (global_step % args.validation_steps == 0)) or global_step == total_training_steps:

                    logger.info("\n\n***** Running evaluation of step {} on dev set *****".format(global_step))
                    val_metrics, best_values, best_steps, predictions = validate(args, model, val_dataloader, tb_writer,
                                                                                 best_values, best_steps, global_step, fairrmetric=fairrmetric)
                    if len(best_steps) and (best_steps[0] == global_step):  # if achieved best performance so far
                        best_metrics = val_metrics.copy()
                        try:
                            freqs_best, _ = utils.plot_rank_barplot(val_dataloader.dataset.reformatted_qrels, predictions,
                                                                    base_pred_scores=(freqs_orig, bins_orig),
                                                                    relevance_thr=args.eval_relevant_at_level,
                                                                    save_as=os.path.join(args.output_dir, "best_rank_historgram.pdf"))
                        except:
                            logger.error('Histogram not possible!')
                        # Export histogram of ranks of relevant documents
                        utils.write_columns_to_csv(os.path.join(args.output_dir, "rank_histogram.csv"),
                                                   columns=(bins_orig, freqs_orig, freqs_best),
                                                   header=('rank', 'Original', 'Reranked'))
                        # Export predictions per query and document
                        ranked_filepath = os.path.join(args.pred_dir, 'best_ranked_top1000.dev.' + args.write_predictions)
                        utils.write_predictions(ranked_filepath, predictions, format=args.write_predictions)
                    metrics_names, metrics_values = zip(*val_metrics.items())
                    running_metrics.append(list(metrics_values))

                    if args.reduce_on_plateau:
                        ROP_scheduler.step(val_metrics[args.reduce_on_plateau])

                    if trial is not None:  # used for hyperparameter optimization
                        trial.report(best_metrics[args.key_metric], global_step)
                        HARD_PATIENCE = 60000  # TODO: Deactivated! int(1e10)
                        # HARD_TOLERANCE = 0.001
                        if len(best_steps) and ((global_step - best_steps[-1]) > HARD_PATIENCE):  # countdown
                        #if (global_step > HARD_PATIENCE) and (running_metrics[0][metric2ind[args.key_metric]] - best_metrics[args.key_metric]) < HARD_TOLERANCE):
                            logger.info('Hard patience reached, stopping training')
                            should_stop = True

                        if trial.should_prune():  # early stopping
                            raise optuna.exceptions.TrialPruned()

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
    _ = utils.export_performance_metrics(metrics_filepath, running_metrics, header, sheet_name="metrics")

    # Export record metrics to a file accumulating records from all experiments
    utils.register_record(args.records_file, args.initial_timestamp, args.experiment_name,
                          best_metrics, val_metrics, comment=args.comment)

    avg_batch_time = batch_times.total_time / (epoch_idx * epoch_steps + step + 1)
    logger.info("Average time to train on 1 batch ({} samples): {:.6f} sec"
                " ({:.6f}s per sample)".format(train_batch_size, avg_batch_time, avg_batch_time / train_batch_size))
    logger.info("Average time to train on 1 batch ({} samples): {:.6f} sec".format(train_batch_size, batch_times.get_average()))
    logger.info('Best {} was {}. Other metrics: {}'.format(args.key_metric, best_values[0], best_metrics))

    logger.debug("Average lookup time: {} s".format(lookup_times.get_average()))
    logger.debug("Average retr. candidates time: {} s".format(retrieve_candidates_times.get_average()))
    logger.debug("Average prep. docids time: {} s".format(prep_docids_times.get_average()))
    logger.debug("Average sample fetching time: {} s".format(sample_fetching_times.get_average()))
    logger.debug("Average collation time: {} s".format(collation_times.get_average()))

    logger.info("Current memory usage: {} MB".format(int(np.round(utils.get_current_memory_usage()))))
    logger.info("Max memory usage: {} MB".format(int(np.ceil(utils.get_max_memory_usage()))))

    return best_metrics


def validate(args, model, val_dataloader, tensorboard_writer, best_values, best_steps, global_step, fairrmetric=None):
    """Run an evaluation on the validation set while logging metrics, and handle result"""

    model.eval()
    eval_start_time = time.time()
    val_metrics, predictions = evaluate(args, model, val_dataloader, fairrmetric=fairrmetric)
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
    if args.key_metric == 'F1_fairness':
        metric_value = 2 * val_metrics['MRR@10'] * val_metrics['NFaiRR_cutoff_10']/(val_metrics['MRR@10'] + val_metrics['NFaiRR_cutoff_10'])
    else:
        metric_value = val_metrics[args.key_metric]
    if args.key_metric in NEG_METRICS:
        ind = bisect.bisect_right(best_values, metric_value)  # index where to insert in sorted list in ascending order
    else:
        ind = utils.reverse_bisect_right(best_values, metric_value)  # index where to insert in sorted list in descending order

    condition = (ind < args.num_keep_best) and (global_step > STEP_THRESHOLD)  # NOTE: the second condition is because fairness is initially always bad but performance at its best
    if condition:
        best_values.insert(ind, metric_value)
        best_steps.insert(ind, global_step)
        # to save space: optimizer, scheduler not saved! Only latest checkpoints can be used for resuming training perfectly
        utils.save_model(os.path.join(args.save_dir, 'model_best_{}.pth'.format(global_step)), global_step, model)
        if len(best_values) > args.num_keep_best:
            os.remove(os.path.join(args.save_dir, 'model_best_{}.pth'.format(best_steps[-1])))
            best_values = best_values[:args.num_keep_best]
            best_steps = best_steps[:args.num_keep_best]

        # Export metrics to a file accumulating best records from the current experiment
        rec_filepath = os.path.join(args.pred_dir, 'training_session_records.xls')
        utils.register_record(rec_filepath, args.initial_timestamp, args.experiment_name, val_metrics)

    return val_metrics, best_values, best_steps, predictions


def evaluate(args, model, dataloader, fairrmetric=None):
    """
    Evaluate a given model on the dataset contained in the given dataloader and compile a dataframe with
    document ranks and scores for each query. If the dataset includes relevance labels (qrels), then metrics
    such as MRR, MAP etc will be additionally computed.
    :return:
        eval_metrics: dict containing metrics (at least 1, batch processing time)
        scores: dict {str(qID): dict{str(passageid): rel_score}}, predicted relevance scores per query and document
    """
    qrels = dataloader.dataset.qrels  # dict{qID: dict{pID: relevance}}
    labels_exist = qrels is not None

    # num_docs is the (potentially variable) number of candidates per query
    relevances = []  # (total_num_queries) list of (num_docs) lists with non-zeros at the indices corresponding to actually relevant passages
    num_relevant = []  # (total_num_queries) list of number of ground truth relevant documents per query
    scores = {}  # dict {str(qID): dict{str(passageid): rel_score}}  # string IDs required for pytrec_eval
    query_time = 0  # average time for the model to score candidates for a single query
    total_loss = 0  # total loss over dataset

    with torch.no_grad():
        for batch_data, qids, docids in tqdm(dataloader, desc="Evaluating"):
            batch_data = {k: v.to(args.device) for k, v in batch_data.items()}
            start_time = time.perf_counter()
            try:
                out = model(**batch_data)
            except RuntimeError:
                raise optuna.exceptions.TrialPruned()
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
                # len(batch_size) list of lists: docIDs and scores per query, in order of descending relevance score
                ranksorted_docs, sorted_scores = zip(*(map(rank_docs, docids, rel_scores)))

            # extend results dictionary by batch_size elements. Indexing works both for np.arrays and list of lists formats for _docs and_scores
            scores.update({str(qids[i]): {str(ranksorted_docs[i][j]): sorted_scores[i][j] for j in range(len(ranksorted_docs[i]))} for i in range(len(qids))})

            if labels_exist:
                relevances.extend(get_relevances(qrels[qids[i]], ranksorted_docs[i], relevant_at_level=dataloader.dataset.relevant_at_level) for i in range(len(qids)))
                # number of g.t. positives in entire dataset for each query
                num_relevant.extend(len([candid for candid in qrels[qid]
                                         if qrels[qid][candid] >= dataloader.dataset.relevant_at_level]) for qid in qids)

    if labels_exist:
        # try:
        # eval_metrics = get_retrieval_metrics(scores, qrels=dataloader.dataset.reformatted_qrels, cutoff_values=(5, args.metrics_k),
        #                                          relevance_level=dataloader.dataset.relevant_at_level, verbose=False)  # {metric_name: metric_value}
        eval_metrics = calculate_metrics(relevances, num_relevant, args.metrics_k) # aggregate metrics for the entire dataset
        # except Exception as x:
        #     print(x)
        #     traceback.print_stack()
        #     ipdb.post_mortem()
        #     logger.error('Metrics calculation failed!')
        #     eval_metrics = OrderedDict()
        eval_metrics['loss'] = total_loss / len(dataloader.dataset)  # average over samples
    else:
        eval_metrics = OrderedDict()
    eval_metrics['query_time'] = query_time / len(dataloader.dataset)  # average over samples

    # Evaluate fairness  # TODO: split into separate function
    if fairrmetric is not None:
        try:
            eval_FaiRR, eval_NFaiRR = fairrmetric.calc_FaiRR_retrievalresults(retrievalresults=scores)

            for _cutoff in eval_FaiRR:
                eval_metrics['NFaiRR_cutoff_%d' % _cutoff] = eval_NFaiRR[_cutoff]
        except:
            logger.error('Fairness metrics calculation failed!')

    return eval_metrics, scores

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
                try:
                    outputs = model(**batch)  # Tuple(similarities, query_embeddings, doc_embeddings)
                except RuntimeError:
                    raise optuna.exceptions.TrialPruned()
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


def get_dataset(args, eval_mode, tokenizer):
    """Initialize and return evaluation dataset object based on args"""

    if args.model_type == 'repbert':
        raise NotImplementedError('RepBERT evaluation is no longer supported. To restore backward compatibility, fix MSMARCODataset.')
        return MSMARCODataset(eval_mode, args.msmarco_dir, args.collection_memmap_dir, args.tokenized_path,
                              args.max_query_length, args.max_doc_length, limit_size=args.eval_limit_size)
    else:
        return MYMARCO_Dataset(eval_mode, args.embedding_memmap_dir, args.eval_query_tokens_path,
                               args.eval_candidates_path, qrels_path=args.eval_qrels_path, query_ids_path=args.eval_query_ids,
                               tokenizer=tokenizer, max_query_length=args.max_query_length,
                               num_candidates=None,  # Always use ALL candidates for evaluation
                               limit_size=args.eval_limit_size,
                               load_collection_to_memory=args.load_collection_to_memory,
                               inject_ground_truth=args.inject_ground_truth,
                               relevance_labels_mapping=args.relevance_labels_mapping,
                               include_at_level=args.eval_include_at_level, relevant_at_level=args.eval_relevant_at_level)


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

        return CODER(custom_encoder=query_encoder,
                     d_model=args.d_model,
                     num_heads=args.num_heads,
                     num_decoder_layers=args.num_layers,
                     dim_feedforward=args.dim_feedforward,
                     dropout=args.dropout,
                     activation=args.activation,
                     normalization=args.normalization_layer,
                     doc_emb_dim=doc_emb_dim,
                     scoring_mode=args.scoring_mode,
                     temperature=args.temperature,
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
    os.makedirs(output_dir, exist_ok=True)

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)
    logger.info("Stored configuration file in '{}'".format(output_dir))

    # Rest of configuration (does not need to be saved as a JSON config file)
    info_dict = {'initial_timestamp': formatted_timestamp,
                 'output_dir': output_dir,
                 'save_dir': os.path.join(output_dir, 'checkpoints'),
                 'pred_dir': os.path.join(output_dir, 'predictions'),
                 'tensorboard_dir': os.path.join(output_dir, 'tb_summaries')}
    config.update(info_dict)

    args = utils.dict2obj(config)  # Convert config dict to args object

    # Create subdirectories
    utils.create_dirs([args.save_dir, args.pred_dir, args.tensorboard_dir])

    # Setup CUDA, GPU
    args.n_gpu = len(args.cuda_ids)
    if torch.cuda.is_available():
        if args.n_gpu > 1:
            args.device = torch.device("cuda")
        else:
            args.device = torch.device("cuda:%d" % args.cuda_ids[0])
    else:
        args.device = torch.device("cpu")

    # Add file logging besides stdout
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'output.log'))
    logger.addHandler(file_handler)

    # Log current hardware setup
    logger.info("Device: %s, n_gpu: %s", args.device, args.n_gpu)
    if args.device.type == 'cuda':
        logger.info("Device: {}".format(torch.cuda.get_device_name(0)))
        total_mem_mb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
        total_mem_mb = int(np.ceil(total_mem_mb))
        logger.info("Total memory: {} MB".format(total_mem_mb))

    # Store additional configuration, code version and environment info
    info_dict['device'] = str(args.device)
    info_dict['n_gpu'] = args.n_gpu
    if args.device.type == 'cuda':
        info_dict['cuda device'] = torch.cuda.get_device_name(0)
        info_dict['cuda memory'] = total_mem_mb

    info_dict['Code path'] = os.path.abspath(__file__)
    info_dict['Git hash'] = utils.get_git_revision_short_hash()
    git_diff = utils.get_git_diff()
    if len(git_diff) > 0:
        info_dict['Git diff'] = "See file 'git_diff.txt'"
        with open(os.path.join(args.output_dir, 'git_diff.txt'), 'w', encoding="utf-8") as f:
            f.write(git_diff)
    else:
        info_dict['Git diff'] = "No changes"
    info_dict['experiment_name'] = args.experiment_name

    utils.write_columns_to_csv(os.path.join(output_dir, "info.txt"), rows=info_dict.items(), delimiter='\t')

    utils.write_conda_env(os.path.join(output_dir, "conda_env.yml"))

    return args


def main(args, trial=None):  # trial is an Optuna hyperparameter optimization object

    if args.debug:
        logger.setLevel('DEBUG')

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv)))  # command used to run the script


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
    utils.set_seed(args.seed, has_gpu=args.n_gpu > 0)

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
    logger.info("'{}' data loaded in {:.3f} sec".format(eval_mode, time.time() - start_time))
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

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
    else:  # new configuration setup for CODER models
        model = get_model(args, eval_dataset.emb_collection.embedding_vectors.shape[1])

    logger.debug("Model:\n{}".format(model))
    logger.info("Total number of model parameters: {}".format(utils.count_parameters(model)))
    logger.info("Total trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))
    logger.info("Number of encoder parameters: {}".format(utils.count_parameters(model.encoder)))
    logger.info("Trainable encoder parameters: {}".format(utils.count_parameters(model.encoder, trainable=True)))

    model.to(args.device)  # will also print model architecture, besides moving to GPU

    collate_fn = eval_dataset.get_collate_func(n_gpu=args.n_gpu, label_format=model.loss_module.formatting)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                 num_workers=args.data_num_workers, collate_fn=collate_fn)

    if args.task == "train":
        return train(args, model, eval_dataloader, tokenizer, fairrmetric=fairrmetric, trial=trial)
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
            eval_metrics, predictions = inspect(args, model, eval_dataloader)
        else:
            eval_metrics, predictions = evaluate(args, model, eval_dataloader, fairrmetric=fairrmetric)
        eval_runtime = time.time() - eval_start_time
        logger.info("Evaluation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))
        print()
        print_str = 'Evaluation Summary: '
        for k, v in eval_metrics.items():
            print_str += '{}: {:8f} | '.format(k, v)
        logger.info(print_str)

        try:
            freqs, bins = utils.plot_rank_barplot(eval_dataloader.dataset.qrels, pred_scores=predictions, save_as=os.path.join(args.pred_dir, "rank_historgram.pdf"))
        except:
            logger.error('Histogram not possible!')

        if args.inject_ground_truth:
            logger.warning("Ground truth documents were injected among candidates! This may cause inflated metrics!")

        filename = 'reranked_' + os.path.basename(args.eval_candidates_path)
        if not filename.endswith('.tsv'):
            filename = filename[:filename.find('_memmap')] + '.tsv'  # it "eats" the last character if not '_memmap'
        ranked_filepath = os.path.join(args.pred_dir, filename)
        logger.info("Writing predicted ranking to: {} ...".format(ranked_filepath))
        utils.write_predictions(ranked_filepath, predictions, format=args.write_predictions)

        # Export record metrics to a file accumulating records from all experiments
        utils.register_record(args.records_file, args.initial_timestamp, args.experiment_name,
                              eval_metrics, comment=args.comment)

        # Export histogram of ranks of relevant documents to a file
        utils.write_columns_to_csv(os.path.join(args.output_dir, "rank_histogram.csv"),
                                   columns=(bins, freqs),
                                   header=('rank', 'Counts'))

        return eval_metrics


if __name__ == "__main__":
    total_start_time = time.time()
    args = run_parse_args()
    args = setup(args)  # Setup experiment session
    main(args)
    logger.info("All done!")
    total_runtime = time.time() - total_start_time
    logger.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))
