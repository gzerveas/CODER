import argparse

NEG_METRICS = ['loss']  # metrics which are better when smaller
POS_METRICS = ['MRR', 'MAP', 'Recall', 'nDCG']  # metrics which are better when higher
POS_METRICS.extend(m + '@' for m in POS_METRICS[:])
METRICS = NEG_METRICS + POS_METRICS


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
    parser.add_argument("--task", choices=["train", "dev", "eval"])
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
    parser.add_argument("--tokenized_path", type=str, default="repbert/preprocessed",
                        help="Contains pre-tokenized/numerized queries in JSON format. Can be dir or file.")
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
    parser.add_argument('--gpu_id', action='store', dest='cuda_ids', type=str, default="0",
                        help="Optional cuda device ids for single/multi gpu setting, like '0' or '0,1,2,3' ", required=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--data_num_workers", default=0, type=int,
                        help="Number of processes feeding data to model. Default: main process only.")
    parser.add_argument("--num_keep", default=1, type=int,
                        help="How many (latest) checkpoints to keep, besides the best. Can be 0 or more"
                             "The 'best' checkpoint takes ~500MB, each 'latest' checkpoint ~1.6GB.")
    parser.add_argument("--load_collection_to_memory", action='store_true',
                        help="If true, will load entire doc. embedding array as np.array to memory, instead of memmap! "
                        "Needs ~26GB for MSMARCO (~50GB project total), but is faster.")
    parser.add_argument("--no_predictions", action='store_true',
                        help="If true, will not write predictions (ranking of candidates in evaluation set) to disk. "
                             "Used to save storage, e.g. when optimizing hyperparameters. (~300MB for 10k queries)")

    ## Training process
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int)
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps. The model parameters will be updated every this many batches")
    parser.add_argument("--validation_steps", type=int, default=2000,
                        help="Validate every this many training steps (i.e. param. updates); 0 for never.")
    parser.add_argument("--save_steps", type=int, default=2000,
                        help="Save checkpoint every this many training steps (i.e. param. updates); "
                             "0 for no periodic saving (save only at the end)")
    parser.add_argument("--logging_steps", type=int, default=200,
                        help="Log training information (tensorboard) every this many training steps; 0 for never")

    parser.add_argument("--num_epochs", default=1, type=int)
    parser.add_argument('--optimizer', choices={"AdamW", "RAdam"}, default="AdamW", help="Optimizer")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)  # 1e-8
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Peak learning rate used globally")  # 3e-6
    parser.add_argument("--encoder_learning_rate", default=None, type=float,
                        help="Special learning rate for the query encoder. By default same as global learning rate.")
    parser.add_argument("--warmup_steps", default=1000, type=int,
                        help="For the first this many steps, the learning rate is linearly increased up to its maximum")  # 10000
    parser.add_argument("--encoder_warmup_steps", default=None, type=int,
                        help="Special for the query encoder. By default same as global warmup steps.")
    parser.add_argument("--encoder_delay", default=0, type=int,
                        help="At which step to start optimizing the encoder.")
    parser.add_argument("--final_lr_ratio", default=0.01, type=float,
                        help="Proportion of the peak learning rate to retain at the end of training. This refers either"
                             "to scheduled decay or Reducing On Plateau.")
    parser.add_argument('--reduce_on_plateau', default=None, choices=METRICS,
                        help="If a metric is specified with this option, then this metric will be monitored, and in "
                             "case of no improvement within `ROP_patience` steps, the learning rate will be reduced by"
                             "`ROP_factor`.")
    parser.add_argument("--ROP_patience", default=10000, type=int,
                        help="Number of steps after which the learning rate will be reduced in case of no performance"
                             " improvement. Must be higher than `validation_steps` (usually, multiple).")
    parser.add_argument("--ROP_cooldown", default=10000, type=int,
                        help="Number of steps after each learning rate reduction, during which the ROP mechanism is "
                             "inactive.")
    parser.add_argument("--ROP_factor", default=0.5, type=float,
                        help="Multiplicative factor used to reduce learning rate in case of no performance"
                             " improvement.")
    parser.add_argument("--ROP_threshold", default=1e-4, type=float,
                        help="Unsigned threshold used to decide whether metric has improved improvement.")
    parser.add_argument("--ROP_thr_mode", default='rel', choices=['rel', 'abs'],
                        help="Type of threshold used to decide whether metric has improved improvement: "
                             "relative ('rel') or absolute ('abs') improvement.")
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
                        help='What kind of positional encoding to use for the transformer decoder input sequence') # TODO: not implemented
    parser.add_argument('--activation', choices={'relu', 'gelu'}, default='gelu',
                        help='Activation to be used in transformer decoder')
    parser.add_argument('--normalization_layer', choices={'BatchNorm', 'LayerNorm'}, default='BatchNorm',
                        help='Normalization layer to be used internally in the transformer decoder') # TODO: not implemented
    parser.add_argument('--scoring_mode',
                        choices={'raw', 'sigmoid', 'tanh', 'softmax',
                                 'cross_attention', 'cross_attention_sigmoid', 'cross_attention_tanh', 'cross_attention_softmax',
                                 'dot_product'},
                        default='raw', help='Scoring function to map the final embeddings to scores')
    parser.add_argument('--loss_type', choices={'multilabelmargin', 'crossentropy'}, default='multilabelmargin',
                        help='Loss applied to document scores')

    args = parser.parse_args()

    if args.task is None and args.config_filepath is None:
        raise ValueError('Please specify task! (train, dev, eval)')

    # User can enter e.g. 'MRR@', indicating that they want to use the provided metrics_k for the key metric
    components = args.key_metric.split('@')
    if len(components) > 1:
        args.key_metric = components[0] + "@{}".format(args.metrics_k)

    if args.resume and (args.load_model_path is None):
        raise ValueError("You can only use option '--resume' when also specifying a model to load!")

    if args.encoder_learning_rate is None:
        args.encoder_learning_rate = args.learning_rate

    if args.encoder_warmup_steps is None:
        args.encoder_warmup_steps = args.warmup_steps

    if args.scoring_mode.endswith('softmax') ^ (args.loss_type == 'crossentropy'):
        raise ValueError('Cross-entropy loss should be used iff softmax is used for scoring')

    if args.ROP_patience < args.validation_steps:
        raise ValueError('If the patience for "reduce on plataeu" is lower than the validation interval, the learning '
                         'rate will often be decreased erroneously (performance changes are only registered every `validation_steps`).')

    if args.reduce_on_plateau and (args.final_lr_ratio > args.ROP_factor):
        raise ValueError('You have set the final learning rate to be {} of the maximum learning rate, but the ROP_factor'
                         ' used to reduce the learning rate is {}.'.format(args.final_lr_ratio, args.ROP_factor))

    args.cuda_ids = [int(x) for x in args.cuda_ids.split(',')]
    args.n_gpu = len(args.cuda_ids)

    return args
