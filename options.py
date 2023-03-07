import argparse
import logging
logging.basicConfig(format='%(asctime)s | %(name)-8s - %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger()


NEG_METRICS = ['loss']  # metrics which are better when smaller
POS_METRICS = ['MRR', 'MAP', 'Recall', 'nDCG', 'F1_fairness']  # metrics which are better when higher
POS_METRICS.extend(m + '@' for m in POS_METRICS[:])
METRICS = NEG_METRICS + POS_METRICS


def run_parse_args():
    parser = argparse.ArgumentParser(description='Run a complete training or evaluation. Optionally, a JSON configuration '
                                                 'file can be used, to overwrite command-line arguments.')
    ## Run from config file
    parser.add_argument('--config', dest='config_filepath',
                        help='Configuration .json file (optional, typically *instead* of command line arguments). '
                             'Existing options inside this file overwrite existing command-line args and defaults, '
                             'except for the ones defined by `override`!')
    parser.add_argument('--override', type=str,
                        help="Optional, to be used with `config`. A string in the format of a python dict, with keys the options "
                             "which should be overriden in the configuration .json file, e.g. {'task':'inspect'}")
    ## Experiment
    parser.add_argument('--name', dest='experiment_name', type=str, default='',
                        help='A string identifier/name for the experiment to be run '
                             '- it will be appended to the output directory name, before the timestamp')
    parser.add_argument('--comment', type=str, default='', help='A comment/description of the experiment')
    parser.add_argument('--no_timestamp', action='store_true',
                        help='If set, a timestamp and random suffix will not be appended to the output directory name')
    parser.add_argument("--task", choices=["train", "dev", "eval", "inspect"], default=None,
                        help="'train' is used to train the model (and validate on a validation set specified by `eval_candidates_path`).\n"
                             "'dev' is used for evaluation when labels are available: this allows to calculate metrics, "
                             "plot histograms, inject ground truth relevant document in set of candidates to be reranked.\n"
                             "'eval' mode is used for evaluation/inference ONLY if NO labels are available.\n"
                             "'inspect' is used to interactively examine an existing ranked candidates file/memmap "
                             "specified by `eval_candidates_path`, together with "
                             "the respective original queries and documents, reconstructed tokenizations, embeddings, "
                             "ground truth relevant documents, and a (raw .tsv) reference ranked candidates file, "
                             "which may or may not correspond to the same ranking as `eval_candidates_path`")
    parser.add_argument('--resume', action='store_true',
                        help='Used together with `load_model`. '
                             'If set, will load `start_step` and state of optimizer, scheduler besides model weights.')

    ## I/O
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Root output directory. Must exist. Time-stamped directories will be created inside.')
    parser.add_argument("--qrels_path", type=str,
                        help="Path of the text file or directory with the ground truth relevance labels, qrels")
    parser.add_argument("--train_candidates_path", type=str, default="~/data/MS_MARCO/BM25_top1000.in_qrels.train.tsv",
                        help="Text file of candidate (retrieved) documents/passages per query. This can be produced by e.g. Anserini."
                             " If not provided, candidates will be sampled at random from the entire collection.")
    parser.add_argument("--eval_candidates_path", type=str, default="~/data/MS_MARCO/BM25_top1000.in_qrels.dev.tsv",
                        help="Text file of candidate (retrieved) documents/passages per query. This can be produced by e.g. Anserini."
                             " If not provided, candidates will be sampled at random from the entire collection.")
    parser.add_argument("--embedding_memmap_dir", type=str, default="repbert/representations/doc_embedding",
                        help="Directory containing (num_docs_in_collection, doc_emb_dim) memmap array of document "
                             "embeddings and an accompanying (num_docs_in_collection,) memmap array of doc/passage IDs")
    parser.add_argument("--tokenized_path", type=str, default="repbert/preprocessed",  # TODO: rename "train_query_tokens_path"
                        help="Contains pre-tokenized/numerized queries in JSON format. Can be dir or file. "
                             "If dir, it should contain: queries.train.json. Otherwise (i.e. if a file), "
                             "it will be used for training and `eval_query_tokens_path` should also be set.")
    parser.add_argument("--eval_query_tokens_path", type=str, default=None,
                        help="Contains pre-tokenized/numerized eval queries in JSON format. When not used, "
                             "`tokenized_path` should be a *directory* containing: queries.{train,dev,eval}.json")
    parser.add_argument("--raw_queries_path", type=str,
                        help="Optional: .tsv file which contains raw text queries (ID <tab> text). Used only for 'inspect' mode.")
    parser.add_argument("--query_emb_memmap_dir", type=str, default="repbert/representations/doc_embedding",
                        help="Optional: Directory containing (num_queries, query_emb_dim) memmap array of query "
                             "embeddings and an accompanying (num_queries,) memmap array of query IDs. Used only for 'inspect' mode")
    parser.add_argument("--raw_collection_path", type=str,
                        help=".tsv file which contains raw text documents (ID <tab> text). Used only for 'inspect' mode.")
    parser.add_argument("--collection_memmap_dir", type=str,
                        help="Optional: Memmap dir containing token IDs for each collection document. RepBERT or 'inspect' mode only!")  # RepBERT/inspect only
    parser.add_argument('--train_query_ids', type=str,
                        help="Optional: Path to a file containing the query IDs to be used for training, each as the first field in each line. "
                             "When not specified, the IDs in `train_candidates_path` are used.")
    parser.add_argument('--eval_query_ids', type=str,
                        help="Optional: Path to a file containing the query IDs to be used for evaluation, each as the first field in each line.. "
                             "When not specified, the IDs in `eval_candidates_path` are used.")
    parser.add_argument('--records_file', default='./records.xls', help='Excel file keeping best records of all experiments')
    parser.add_argument('--load_model', dest='load_model_path', type=str,
                        help='Path to pre-trained model checkpoint. If specified, the model weights will be loaded, BUT NOT the state'
                             'of the optimizer/scheduler. To resume training, additionally set the flag `--resume`.')
    # The following are currently used only if `model_type` is NOT 'repbert'
    parser.add_argument("--query_encoder_from", type=str, default="bert-base-uncased",
                        help="""A string used to initialize the query encoder weights and config object: 
                        can be either a pre-defined HuggingFace transformers string (e.g. "bert-base-uncased"), or
                        a path of a directory containing weights and config file.""")
    parser.add_argument("--query_encoder_config", type=str, default=None,
                        help="""Optional: A string used to define the query encoder configuration.
                        Used for flexibility, in case only the weights should be initialized by `query_encoder_from`. 
                        Can be either a pre-defined HuggingFace transformers string (e.g. "bert-base-uncased"), or
                        a path of a directory containing the config file, or directly the JSON config path.""")
    parser.add_argument("--tokenizer_from", type=str, default=None,
                        help="""Optional: Path to a directory containing a saved custom tokenizer (vocabulary and added tokens)
                        for queries, or a HuggingFace built-in string. Only used if for whatever reason 
                        the query tokenizer should differ from what is specified by `query_encoder_from` and `query_encoder_config`""")
    

    ## Dataset
    parser.add_argument('--train_limit_size', type=float, default=None,
                        help="Limit  dataset to specified smaller random sample, e.g. for debugging purposes. "
                             "If in [0,1], it will be interpreted as a proportion of the dataset, "
                             "otherwise as an integer absolute number of samples")
    parser.add_argument('--eval_limit_size', type=float, default=None,
                        help="Limit  dataset to specified smaller random sample, e.g. for debugging purposes. "
                             "If in [0,1], it will be interpreted as a proportion of the dataset, "
                             "otherwise as an integer absolute number of samples")
    parser.add_argument("--max_query_length", type=int, default=32,
                        help="Number of tokens to keep from each query.")
    parser.add_argument("--max_doc_length", type=int, default=256,  # RepBERT/inspect only
                        help="Optional: Number of tokens to keep from each document. Used for RepBERT or 'inspect' mode only")
    parser.add_argument('--num_candidates', type=int, default=None,
                        help="When training, the number of document IDs to sample from all document IDs corresponding to a query and found"
                             " in `train_candidates_path` file. If None, all found document IDs will be used. "
                             "If no `train_candidates_path` is provided, negatives will be sampled from the entire collection"
                             " at random, and therefore `num_candidates` in this case CANNOT be None. When evaluating, "
                             "this is always None (i.e. all candidates in `eval_candidates_path` are used).")
    parser.add_argument('--num_random_neg', type=int, default=0,
                        help="Number of negatives to randomly sample from other queries in the batch for training. "
                             "If 0, only documents in `train_candidates_path` will be used as negatives.")
    parser.add_argument('--include_at_level', default=1,
                        help="The relevance score that candidates in `qrels_path` should at least have (after mapping) "
                             "in order to be included in the set of candidate documents (as positives/negatives). "
                             "Typically only set to 0 if one expects these to be reliable negatives.")
    parser.add_argument('--relevant_at_level', default=1,
                        help="The relevance score that candidates in `qrels_path` should at least have (after mapping) "
                             "in order to be considered relevant in training and evaluation (incl. metrics calculation)."
                             " Below this level, the target relev. prob. will be 0."
                             "Should be at least as high as `include_at`, and is typically > 0.")
    parser.add_argument("--relevance_labels_mapping", type=str, default=None,
                        help="Optional: A string used to define a dictionary used to override/map relevance scores as "
                             "given in `qrels_path` to a new value, e.g. {1: 0.333}")

    ## System
    parser.add_argument('--debug', action='store_true', help="Activate debug mode (displays more information)")
    parser.add_argument('--gpu_id', action='store', dest='cuda_ids', type=str, default="0",
                        help="Optional cuda device ids for single/multi gpu setting, like '0' or '0,1,2,3' ", required=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--data_num_workers", default=0, type=int,
                        help="Number of processes feeding data to model. Default: main process only.")
    parser.add_argument("--num_keep", default=1, type=int,
                        help="How many (latest) checkpoints to keep, besides the best. Can be 0 or more. "
                             "Each 'best' checkpoint takes ~500MB (BERT-base), each 'latest' checkpoint ~1.6GB.")
    parser.add_argument("--num_keep_best", default=1, type=int,
                        help="How many best checkpoints to keep. Can be 1 or more. "
                             "Each 'best' checkpoint takes ~500MB (BERT-base), each 'latest' checkpoint ~1.6GB.")
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
                             "`ROP_factor`, until `final_lr_ratio` is reached")
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
    parser.add_argument("--inject_ground_truth", action='store_true',
                        help="If true, the ground truth document(s) will always be present in the set of documents"
                             "to be reranked for evaluation (during validation or in 'dev' mode).")

    ## Model
    parser.add_argument("--model_type", type=str, choices=['repbert', 'mdstransformer'], default='mdstransformer',
                        help="""Type of the entire (end-to-end) information retrieval model""")
    parser.add_argument("--query_aggregation", type=str, choices=['mean', 'first'], default='mean',
                        help="""How to aggregate the individual final encoder embeddings corresponding to query tokens into 
                        a single vector.""")
    # parser.add_argument('--token_type_ids', action='store_true',
    #                     help="If set, a tensor of 0s will be passed to the HuggingFace query encoder as an input "
    #                          "for 'token_type_ids'. However, this is the HuggingFace default for BERT, so it is unnecessary.")

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
    parser.add_argument('--normalization_layer', choices={'BatchNorm', 'LayerNorm', 'None'}, default='LayerNorm',
                        help='Normalization layer to be used internally in the transformer decoder')
    parser.add_argument('--selfatten_mode', type=int, default=0, choices=[0, 1, 2, 3],
                        help="Self-attention (SA) mode for contextualizing documents. Choices: "
                             "0: regular SA "
                             "1: turn off SA by using diagonal SA matrix (no interactions between documents) "
                             "2: linear layer + non-linearity instead of SA "
                             "3: SA layer has simply been removed")
    parser.add_argument('--no_dec_crossatten', action='store_true',
                        help="If used, the transformer decoder will not have cross-attention over the sequence of "
                             "query term embeddings in the output of the query encoder")
    parser.add_argument('--no_decoder', action='store_true',
                        help="If used, no transformer decoder will be used to transform document embeddings")
    # Scoring and loss
    parser.add_argument('--scoring_mode',
                        choices={'raw', 'sigmoid', 'tanh', 'softmax',
                                 'cross_attention', 'cross_attention_sigmoid', 'cross_attention_tanh', 'cross_attention_softmax',
                                 'dot_product', 'dot_product_gelu', 'dot_product_softmax',
                                 'cosine', 'cosine_gelu', 'cosine_softmax'},
                        default='raw', help='Scoring function to map the final embeddings to scores')
    parser.add_argument('--loss_type', choices={'multilabelmargin', 'crossentropy', 'listnet', 'multitier'},
                        default='multilabelmargin', help='Loss applied to document scores')
    parser.add_argument('--aux_loss_type', choices={'multilabelmargin', 'crossentropy', 'listnet', 'multitier', None},
                        default=None,
                        help='Auxiliary loss (optional). If specified, it will be multiplied by `aux_loss_coeff` and '
                             'added to the main loss.')
    parser.add_argument('--aux_loss_coeff', type=float, default=0,
                        help="Coefficient of auxiliary loss term specified by `aux_loss_type`.")
    parser.add_argument('--num_tiers', type=int, default=4,
                        help="Number of relevance tiers for `loss_type` 'multitier'. "
                             "Ground truth documents are not considered a separate tier.")
    parser.add_argument('--tier_size', type=int, default=50,
                        help="Number of candidates within each tier of relevance for `loss_type` 'multitier'")
    parser.add_argument('--tier_distance', type=int, default=None,
                        help="Number of candidates separating (as a buffer) tiers of relevance for `loss_type` 'multitier'. "
                             "If None, the distance will be automatically calculated so as to place the tier centers "
                             "as widely apart as possible.")
    parser.add_argument('--diff_function', choices={'exp', 'maxmargin'}, default='maxmargin',
                        help="Function to be applied to score differences between documents belonging to different tiers, "
                             "when `loss_type` is 'multitier'.")
    parser.add_argument('--gt_function', choices={'multilabelmargin', 'same', None}, default=None,
                        help="Special loss function to be applied for calculating the extra contribution of the ground truth"
                             "relevant documents, when `loss_type` is 'multitier'. If None, no special treatment will "
                             "be given to ground truth relevant documents in the loss calculation, besides including them "
                             "in the top tier.")
    parser.add_argument('--gt_factor', type=float, default=2.0,
                        help="Scaling factor of ground truth component for `loss_type` 'multitier'")

    ## Debiasing (bias/neutrality regularization)
    parser.add_argument("--collection_neutrality_path", type=str,
                        help="path to the file containing neutrality values of documents in tsv format (docid [tab] score)")
    parser.add_argument("--background_set_runfile_path", type=str, 
                        default="fair_retrieval/resources/msmarco_fair.background_run.txt",
                        help="path to the TREC run file containing the documents of the background set")
    parser.add_argument('--bias_regul_coeff', type=float, default=0.0,
                        help='Coefficient of the bias term added to loss')
    parser.add_argument('--bias_regul_cutoff', type=int, default=100,
                        help='Bias term is calculated according to the top-X predicted results')

    args = parser.parse_args()

    return args


def check_args(config):
    """Check validity of settings and make necessary conversions"""

    if config['task'] is None:
        raise ValueError('Please specify task! (train, dev, eval, inspect)')

    if config['task'] == 'inspect':
        # config['inject_ground_truth'] = False
        # logger.info("Ground truth documents will not be injected among candidates, because `task == 'inspect'`")
        if config['raw_queries_path'] is None:
            logger.warning("You must set `raw_queries_path` to inspect original queries.")
        if config['raw_collection_path'] is None:
            logger.warning("You must set `raw_collection_path` to inspect original documents.")

    if config['eval_query_tokens_path'] is None:
        config['eval_query_tokens_path'] = config['tokenized_path']

    # User can enter e.g. 'MRR@', indicating that they want to use the provided metrics_k for the key metric
    components = config['key_metric'].split('@')
    if len(components) > 1:
        config['key_metric'] = components[0] + "@{}".format(config['metrics_k'])

    if config['resume'] and (config['load_model_path'] is None):
        raise ValueError("You can only use option '--resume' when also specifying a model to load!")

    if (config['load_model_path'] is not None) and not config['resume']:
        logger.warning("Only model weights will be loaded from '{}'. "
                       "Set option `--resume` if you wish to restore optimizer/scheduler.".format(config['load_model_path']))

    if config['encoder_learning_rate'] is None:
        config['encoder_learning_rate'] = config['learning_rate']

    if config['encoder_warmup_steps'] is None:
        config['encoder_warmup_steps'] = config['warmup_steps']

    if config['scoring_mode'].endswith('softmax') ^ (config['loss_type'] == 'crossentropy'):
        raise ValueError('Cross-entropy loss should be used iff softmax is used for scoring')

    if config['ROP_patience'] < config['validation_steps']:
        raise ValueError('If the patience for "reduce on plataeu" is lower than the validation interval, the learning '
                         'rate will often be decreased erroneously (performance changes are only registered every `validation_steps`).')

    if config['reduce_on_plateau'] and (config['final_lr_ratio'] > config['ROP_factor']):
        raise ValueError('You have set the final learning rate to be {} of the maximum learning rate, but the ROP_factor'
                         ' used to reduce the learning rate is {}.'.format(config['final_lr_ratio'], config['ROP_factor']))

    if type(config['cuda_ids']) is not list:
        config['cuda_ids'] = [int(x) for x in config['cuda_ids'].split(',')]

    if config['aux_loss_type'] is not None and (config['aux_loss_coeff'] <= 0):
        raise ValueError('You have specified an auxiliary loss, but not a positive `aux_loss_coeff`')

    if config['relevance_labels_mapping'] is not None:
        config['relevance_labels_mapping'] = eval(config['relevance_labels_mapping'])  # convert string to dict

    if config['relevant_at_level'] < config['include_at_level']:
        raise ValueError("'relevant_at_level should be at least as high as 'include_at_level'")

    return config
