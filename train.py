import os
import re

import random
import time
import logging
import argparse
import subprocess

import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler
from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers import BertConfig, BertTokenizer, BertModel, RobertaModel, get_linear_schedule_with_warmup

from modeling import RepBERT_Train, MDSTransformer
from dataset import MSMARCODataset, get_collate_function
from optimizers import get_optimizer_class
import utils

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def run_parse_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task", choices=["train", "dev", "eval"], required=True)
    parser.add_argument("--output_dir", type=str, default=f"./data/train")

    parser.add_argument("--msmarco_dir", type=str, default=f"./data/msmarco-passage")
    parser.add_argument("--collection_memmap_dir", type=str, default="./data/collection_memmap")
    parser.add_argument("--tokenize_dir", type=str, default="./data/tokenize")
    parser.add_argument("--max_query_length", type=int, default=20)
    parser.add_argument("--max_doc_length", type=int, default=256)

    ## Training process
    parser.add_argument("--load_model_path", type=str, default=None)
    parser.add_argument("--per_gpu_eval_batch_size", default=26, type=int, )
    parser.add_argument("--per_gpu_train_batch_size", default=26, type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)

    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--evaluate_during_training", action="store_true")
    parser.add_argument("--training_eval_steps", type=int, default=5000)

    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--data_num_workers", default=0, type=int)

    parser.add_argument('--optimizer', choices={"AdamW", "RAdam"}, default="AdamW", help="Optimizer")
    parser.add_argument("--learning_rate", default=3e-6, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--warmup_steps", default=10000, type=int)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=1, type=int)

    ## Model
    parser.add_argument("--model_type", type=str, choices=['repbert', 'mdstransformer'], default='repbert',
                        help="""Type of the entire (end-to-end) information retrieval model""")
    # The following are currently used only if `model_type` is not 'repbert'
    parser.add_argument("--query_encoder_type", type=str, choices=['bert', 'roberta'], default='bert',
                        help="""Type of the model component used for encoding queries""")
    parser.add_argument("--query_encoder_from", type=str, default="bert-base-uncased",
                        help="""A string used to initialize the query encoder weights and config object: 
                        can be either a pre-defined HuggingFace transformers string (e.g. "bert-base-uncased"), or
                        a path of a directory containing weights and config file""")
    parser.add_argument("--query_encoder_config", type=str, default=None,
                        help="""A string used to define the query encoder configuration (optional): 
                        can be either a pre-defined HuggingFace transformers string (e.g. "bert-base-uncased"), or
                        a path of a directory containing the config file, or directly the JSON config path
                        Used in case only the weights should be initialized by `query_encoder_from`""")

    args = parser.parse_args()

    time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    args.tboard_dir = f"{args.output_dir}/log/{time_stamp}"
    args.model_save_dir = os.path.join(args.output_dir, "checkpoints")
    args.eval_save_dir = f"{args.output_dir}/eval_results"
    return args


def train(args, model, val_dataloader):
    """ Train the model """
    tb_writer = SummaryWriter(args.tboard_dir)

    train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = MSMARCODataset("train", args.msmarco_dir,
                                   args.collection_memmap_dir, args.tokenize_dir,
                                   args.max_query_length, args.max_doc_length)

    # NOTE: Must Sequential! Pos, Neg, Pos, Neg, ...
    # GEO: This is because a (query, pos. doc, neg. doc) triplet is split in 2 consecutive samples: (qID, posID) and (qID, negID)
    # GEO: If random sampling had been chosen, then these 2 samples would have ended up in different batches
    train_sampler = SequentialSampler(train_dataset)
    collate_fn = get_collate_function(mode="train")
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=train_batch_size, num_workers=args.data_num_workers,
                                  collate_fn=collate_fn)

    total_training_steps = (len(train_dataloader) // args.gradient_accumulation_steps) * args.num_train_epochs

    global_step = 0  # counts how many times the weights have been updated, i.e. num. batches // gradient acc. steps

    # Prepare optimizer and schedule (linear warmup and decay)
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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_training_steps)

    # Load model and possibly optimizer/scheduler state
    if args.load_model_path:
        model, start_epoch, optimizer, scheduler = utils.load_model(model, args.load_model_path, optimizer, scheduler,
                                                                    args.resume)
    model.to(args.device)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples: %d", len(train_dataset))
    logger.info("  Num Epochs: %d", args.num_train_epochs)
    logger.info("  Batch size per GPU: %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation): %d",
                train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps: %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps: %d", total_training_steps)

    tr_loss = 0  # this is the training loss accumulated from the beginning of training
    logging_loss = 0  # this is synchronized with `tr_loss` every args.logging_steps
    model.zero_grad()
    epoch_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    utils.set_seed(args)  # Added here for reproducibility
    for epoch_idx, _ in enumerate(epoch_iterator):
        batch_iterator = tqdm(train_dataloader, desc="Batch")
        for step, (model_inp, _, _) in enumerate(batch_iterator):

            model_inp = {k: v.to(args.device) for k, v in model_inp.items()}
            model.train()
            outputs = model(**model_inp)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()  # calculate gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.evaluate_during_training and (global_step % args.training_eval_steps == 0):
                    mrr = evaluate(args, model, val_dataloader, mode="dev", prefix="step_{}".format(global_step))
                    tb_writer.add_scalar('dev/MRR@10', mrr, global_step)
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    cur_loss = (tr_loss - logging_loss) / args.logging_steps  # mean loss over last args.logging_steps (smoothened "current loss")
                    tb_writer.add_scalar('train/loss', cur_loss, global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    utils.save_model(args.model_save_dir, global_step, model, optimizer, scheduler)


def evaluate(args, model, dataloader, mode, prefix):

    if not os.path.exists(args.eval_save_dir):
        os.makedirs(args.eval_save_dir)

    logger.info("***** Running evaluation of {} on {} set *****".format(prefix, mode))

    output_file_path = os.path.join(args.eval_save_dir, prefix + ".{}.score.tsv".format(mode))
    with open(output_file_path, 'w') as outputfile:
        for batch, qids, docids in tqdm(dataloader, desc="Evaluating"):
            model.eval()
            with torch.no_grad():
                batch = {k: v.to(args.device) for k, v in batch.items()}
                outputs = model(**batch)
                scores = torch.diagonal(outputs[0]).detach().cpu().numpy()
                assert len(qids) == len(docids) == len(scores)
                for qid, docid, score in zip(qids, docids, scores):
                    outputfile.write(f"{qid}\t{docid}\t{score}\n")

    rank_output = os.path.join(args.eval_save_dir, prefix + ".{}.rank.tsv".format(mode))
    utils.generate_rank(output_file_path, rank_output)

    if mode == "dev":
        mrr = utils.eval_results(rank_output)
        return mrr


def main():
    args = run_parse_args()

    # Setup CUDA, GPU 
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Log current hardware setup
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    utils.set_seed(args)

    # Initialize model
    if args.model_type == 'repbert':
        # keep configuration setup like RepBERT. The model is a common/shared BERT query-document encoder, without interactions
        # between query and document token representations
        load_model_path = "bert-base-uncased" if args.load_model_path is None else args.load_model_path
        # Works with either directory path containing HF config file, or JSON HF config file,  or pre-defined model string
        config = BertConfig.from_pretrained(load_model_path)
        model = RepBERT_Train.from_pretrained(load_model_path, config=config)
    else:  # new configuration setup for MultiDocumentScoringTransformer models
        model = get_model(args)

    logger.info("Experiment configuration: %s", args)
    print("Model:")
    print(model)
    print("Total number of parameters: {}".format(utils.count_parameters(model)))
    print("Trainable parameters: {}".format(utils.count_parameters(model, trainable=True)))

    # Load evaluation set and initialize evaluation dataloader
    eval_mode = 'dev' if args.task == 'train' else 'eval'  # 'eval' here is the name of the MSMARCO test set, 'dev' is the validation set
    logger.info("Preparing {} dataset ...".format(eval_mode))
    eval_dataset = MSMARCODataset(eval_mode, args.msmarco_dir,
                                  args.collection_memmap_dir, args.tokenize_dir,
                                  args.max_query_length, args.max_doc_length)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    collate_fn = get_collate_function(mode=eval_mode)
    # Note that DistributedSampler samples randomly
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                                 num_workers=args.data_num_workers, collate_fn=collate_fn)

    logger.info("  Number of {} samples: {}".format(eval_mode, len(eval_dataset)))
    logger.info("  Batch size: %d", args.eval_batch_size)


    if args.task == "train":
        train(args, model, eval_dataloader)
    else:
        # Just evaluate on some dataset

        # only composite (non-repbert) models need to be loaded; repbert is already loaded at this point
        if not ((args.load_model_path is None) or (args.model_type == 'repbert')):
            model, global_step, _, _ = utils.load_model(model, args.load_model_path)
        model.to(args.device)
        model_checkpoint_name = os.path.splitext(os.path.basename(args.load_model_path))[0]

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        result = evaluate(args, model, eval_dataloader, args.task, prefix=model_checkpoint_name)
        print(result)


def get_query_encoder(args):
    """Initialize and return query encoder model object based on args"""

    if args.query_encoder_type == 'bert':
        return BertModel.from_pretrained(args.query_encoder_from, config=args.query_encoder_config_from)
    elif args.query_encoder_type == 'roberta':
        return RobertaModel.from_pretrained(args.query_encoder_from, config=args.query_encoder_config_from)


def get_model(args):
    """Initialize and return end-to-end model object based on args"""

    query_encoder = get_query_encoder(args)

    if args.model_type == 'mdstransformer':
        return MDSTransformer(encoder_config=query_encoder,
                              d_model=args.d_model,
                              num_heads=args.num_heads,
                              num_decoder_layers=args.num_decoder_layers,
                              dim_feedforward=args.dim_feedforward,
                              dropout=args.dropout,
                              activation=args.activation)
    else:
        raise NotImplementedError('Unknown model type')


if __name__ == "__main__":
    main()
