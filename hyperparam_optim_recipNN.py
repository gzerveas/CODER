import random
import time
import os

import optuna
from optuna.samplers import TPESampler, GridSampler
from optuna.pruners import MedianPruner

import reciprocal_compute
from reciprocal_compute import recip_NN_rerank, extend_relevances
import options
import main
import utils

# Metric for hyperparam optimization.
# Can be different from "key_metric" of main, which determines the set of "best_values" and saved checkpoints
OPTIM_METRIC = 'loss'  # 'MRR@10', 'nDCG@10'
NEG_METRICS = ['loss']

## Hyperparameter optimization settings ##

task = 'smooth_labels'  # 'rerank', 'smooth_labels', 'smooth_labels_e2e'
dataset = 'MSMARCO'  # 'TripClick', 'MSMARCO'
sim_type = 'mixed'  # applies to training   #trial.suggest_categorical("sim_type", ['jaccard', 'geometric', 'mixed'])

# For tasks involving training

# Different settings depending on target score type
if sim_type == 'jaccard':
    study_name = 'recipNN_smooth_labels_JaccOnly_study'
    exp_name_prefix = 'AUTO_JaccOnly'
    boost_factor_raw_max = 10
    boost_factor_norm_max = 10
elif sim_type == 'geometric':
    study_name = 'smooth_labels_GeomOnly_study'
    exp_name_prefix = 'AUTO_GeomOnly'
    boost_factor_raw_max = 1.1
    boost_factor_norm_max = 3
else:  # mixed
    study_name = 'recipNN_smooth_labels_study'
    exp_name_prefix = 'AUTO_RNN-'
    if task == 'smooth_labels':  # here we need to specify the RNN config; in e2e it is dynamically generated
        if dataset == 'TripClick':
            exp_name_prefix += 'l3b'  # ID of RNN configuration. ZeK (e2e), l3b (hit 1000) is the best config for TripClick
        else:
            exp_name_prefix += 'Kb3'  # mVL (auto-HIT), r67 (hit 1000) is the best TAS-B config for MSMARCO, Kb3 (auto-HIT) is the best CoCoDenser config for MSMARCO
    boost_factor_raw_max = 1.25
    boost_factor_norm_max = 5

RETURN_TOP_e2e = 300  # maximum number of top documents to return for each query when optimizing end-to-end

if dataset == 'MSMARCO':
    exp_name_suffix = '_CODER-CoCo-IZc'  # 'CODER-TASB-IZc'
    if task == 'smooth_labels_e2e':  # for smooth_labels we use the config file specified when launching this script
        train_config_file = "/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/configs/MS_MARCO/SmoothL_Qenc_cocodenser_augsep_cocodenser1000_Rneg0_ListnetLoss_Axm_IZc_config_2022-04-03_23-04-54_BHh.config"
        #train_config_file = '/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/configs/MS_MARCO/boost_factor.config'  # path to config file for training
else:  # TripClick
    if task == 'smooth_labels':
        exp_name_suffix = '_TripClick_RAW_inc0.1_rel1_trainHEADnTORSO_valHEAD_CODER_repbert295k_xAk'  # trainALL
    elif task == 'smooth_labels_e2e':
        exp_name_suffix = '_TripClick_RAW_inc0.1_rel1_trainHEADnTORSO_valHEAD_CODER_repbert295k_xAk'
        train_config_file = '/users/gzerveas/data/gzerveas/RecipNN/smooth_labels/configs/TripClick/AUTOe2e_HEADnTORSO_TripClick_CODER_RepBERT_xAk.config'  # path to config file for training


def recipNN_rerank_objective(trial):
    """Optimizes reciprocal NN reranking hyperparameters using Optuna.
    The Optuna script (this) is launched by a command-line command or config file with options as if intended 
    for reciprocal_compute.py, but the actual main function is called from here."""
    random.seed()  # re-randomize seed (it's fixed inside `main`), because it's needed for dir name suffix
    args = reciprocal_compute.run_parse_args()  # `argsparse` object
    config = utils.load_config(args)  # config dictionary, which potentially comes from a JSON file or command-line overrides
    args = utils.dict2obj(config)  # convert back to args object
    args.config_filepath = None  # the contents of a JSON file (if specified) have been loaded already, so prevent the `setup` from overwriting the Optuna overrides

    ## Optuna overrides
    args.hit = trial.suggest_int('hit', 40, 1000, log=True)  # number of candidates to consider when computing labels
    args.sim_mixing_coef = trial.suggest_float('sim_mixing_coef', 1e-3, 1, log=True)
    args.k = trial.suggest_int('k', 3, 30)
    args.trust_factor = trial.suggest_float('trust_factor', 0, 0.8) #trial.suggest_categorical("trust_factor", [0, 0.5]) # [0, 0.5]
    args.k_exp = trial.suggest_int('k_exp', 1, 20)
    args.normalize = trial.suggest_categorical("normalize", ['max', 'mean', 'None'])
    args.weight_func = 'linear' #trial.suggest_categorical("weight_func", ['exp', 'linear'])
    if args.weight_func == 'exp':
        if args.normalize == 'None':  # NOTE: with arbitrary score values, the exponential parameter can easily lead to Inf!
            args.weight_func_param = trial.suggest_float("weight_func_param", 0.001, 1, log=True)
        else:
            args.weight_func_param = trial.suggest_float("weight_func_param", 1.0, 10.0, log=True)
    else:
        args.weight_func_param = 1.0  #trial.suggest_float("weight_func_param", 1.0, 1.0, log=True)  # for linear, the parameter value doesn't matter


    args = reciprocal_compute.setup(args)  # dump config file, create output dir, etc.

    best_values = recip_NN_rerank(args)  # run main function and get best metrics found during evaluation

    for name, value in best_values.items():
        trial.set_user_attr(name, value)  # log in database / study object

    return best_values[OPTIM_METRIC]


def smooth_labels_e2e_objective(trial):
    """Objective function for end-to-end hyperparameter optimization of recipNN label smooothing and subsequent training.
    The Optuna script (this) is launched by a command-line command or config file with options as if intended 
    for reciprocal_compute.py, whose main function is called from here, followed by the main function for training.
    The latter's options are specified in `train_config_file`.
    """
    random.seed()  # re-randomize seed (it's fixed inside `main`), because it's needed for dir name suffix
    # Setup label smoothing
    args_rNN = reciprocal_compute.run_parse_args()  # parse arguments with `reciprocal_compute.py` parsing function
    config = utils.load_config(args_rNN)  # config dictionary, which potentially comes from a JSON file or command-line overrides
    args_rNN = utils.dict2obj(config)  # convert back to args object
    args_rNN.config_filepath = None  # the contents of a JSON file (if specified) have been loaded already, so prevent the `setup` from overwriting the Optuna overrides

    args_rNN.write_to_file = 'pickle'  # this is how we pass the target scores to `main` function

    # ReciprocalNN hyperparams
    args_rNN.hit = trial.suggest_int('hit', 40, 1000)  # number of candidates to consider when computing labels
    if sim_type == 'geometric':
        args_rNN.sim_mixing_coef = trial.suggest_float('sim_mixing_coef', 1, 1, log=True)
    elif sim_type == 'jaccard':
        args_rNN.sim_mixing_coef = trial.suggest_float('sim_mixing_coef', 0, 0, log=True)
    else:
        args_rNN.sim_mixing_coef = trial.suggest_float('sim_mixing_coef', 1e-3, 1, log=True)
    args_rNN.k = trial.suggest_int('k', 3, 30)
    args_rNN.trust_factor = trial.suggest_categorical("trust_factor", [0, 0.5])
    args_rNN.k_exp = trial.suggest_int('k_exp', 1, 10)
    args_rNN.normalize = trial.suggest_categorical("normalize", ['max', 'mean', 'None'])
    args_rNN.weight_func = trial.suggest_categorical("weight_func", ['linear'])#['exp', 'linear'])
    if args_rNN.weight_func == 'exp':
        if args_rNN.normalize == 'None':  # NOTE: with arbitrary score values, the exponential parameter can easily lead to Inf!
            args_rNN.weight_func_param = trial.suggest_float("weight_func_param", 0.001, 1, log=True)
        else:
            args_rNN.weight_func_param = trial.suggest_float("weight_func_param", 1.0, 10.0, log=True)
    else:
        args_rNN.weight_func_param = trial.suggest_float("weight_func_param", 1.0, 1.0, log=True)  # for linear, the parameter value doesn't matter

    args_rNN.return_top = min(RETURN_TOP_e2e, args_rNN.hit)  # number of candidates per query to return

    # ReciprocalNN label smoothing hyperparams
    args_rNN.rel_aggregation = trial.suggest_categorical("rel_aggregation", ['max', 'mean'])
    # args_rNN.redistribute = trial.suggest_categorical("redistribute", ['radically'])
    # args_rNN.norm_relevances = trial.suggest_categorical("norm_relevances", ['max', 'minmax', 'maxminmax', 'None', 'std'])
    # args_rNN.boost_factor = trial.suggest_float("boost_factor", 1.00, 10.0, log=False)
    ## args_rNN.redistr_prt = trial.suggest_float("redistr_prt", 0.01, 0.8, log=False)

    args_rNN = reciprocal_compute.setup(args_rNN)  # dump config file, create output dir, etc.

    # Run function to recompute relevance labels and write to disk as a file
    _ = extend_relevances(args_rNN)

    del config
    # Read configuration for training
    # parse arguments with `options.py` parsing function, to ensure all arguments expected by `load_config` are present
    args = options.run_parse_args('--config {}'.format(train_config_file).split(' '))

    config = utils.load_config(args)  # config dictionary, which comes from `train_config_file` JSON file
    args = utils.dict2obj(config)  # convert back to args object
    args.target_scores_path = args_rNN.out_filepath  # the file used to store smoothened labels becomes the target labels file for training
    args.config_filepath = None  # the contents of the JSON file have been loaded already, so prevent the `main.setup` from overwriting the Optuna overrides

    args.output_dir = os.path.join(args.output_dir, "AUTOe2e_HEADnTORSO/")  # append to output dir to distinguish from other runs

    # Training Hyperparameters for label smoothing
    args.boost_relevant = "constant"
    args.label_normalization = trial.suggest_categorical("label_normalization", ['maxmin', 'maxminmax', 'std', 'None'])

    if args.label_normalization in {'None', 'maxminmax'}:   # large magnitude boost factors are not needed for these normalizations
        args.boost_factor = trial.suggest_float("boost_factor", 1.0, boost_factor_raw_max, log=True)  # 1.2 max for mixed geometric/Jaccard, 10 for Jaccard only
    else:
        args.boost_factor = trial.suggest_float("boost_factor", 1.0, boost_factor_norm_max, log=True)

    args.max_inj_relevant = trial.suggest_int("max_inj_relevant", 3, RETURN_TOP_e2e, log=True)

    if args.label_normalization == 'None':
        args.label_normalization = None  # the actual expected value in the code is None, not 'None'

    # Other hyperparameters for training
    args.num_epochs = 80
    args.learning_rate = trial.suggest_loguniform('learning_rate', 5e-7, 3e-6)
    args.encoder_learning_rate = args.learning_rate
    args.warmup_steps = int(trial.suggest_discrete_uniform('warmup_steps', 5000, 30000, 1000))
    args.encoder_warmup_steps = args.warmup_steps
    args.final_lr_ratio = trial.suggest_uniform('final_lr_ratio', 0.1, 0.999)

    # Set name of experiment
    exp_name_pref = exp_name_prefix + args_rNN.rand_suffix  # random ID of RNN configuration
    args.experiment_name = exp_name_pref + f'_hit{args_rNN.hit}' + f'_rboost{round(args.boost_factor, 3)}_norm-{args.label_normalization}_top{args.max_inj_relevant}'
    args.experiment_name += f'_lr{args.learning_rate:.2e}_warmup{args.warmup_steps}_finlrr{args.final_lr_ratio:.2f}'
    args.experiment_name += exp_name_suffix

    # Run main function for training
    config = main.setup(args)  # Setup main training session (logging, output dir, dump config file, etc.)
    best_values = main.main(config, trial)  # best metrics found during evaluation

    for name, value in best_values.items():
        trial.set_user_attr(name, value)  # log in database / study object

    return best_values[OPTIM_METRIC]


def smooth_labels_objective(trial):
    """Objective function for hyperparameter optimization of training with smooth labels.
    The Optuna script (this) is launched with a config file with options as if intended for main.py"""
    random.seed()  # re-randomize seed (it's fixed inside `main`), because it's needed for dir name suffix

    # Run main function hyperparam optimization for training
    args = options.run_parse_args()  # `argsparse` object for training

    # Optuna overrides for main training
    config = utils.load_config(args)  # config dictionary, which potentially comes from a JSON file
    args = utils.dict2obj(config)  # convert back to args object
    args.config_filepath = None  # the contents of a JSON file (if specified) have been loaded already, so prevent the `main.setup` from overwriting the Optuna overrides

    # Hyperparameters for label smoothing training
    args.boost_relevant = "constant"
    args.label_normalization = trial.suggest_categorical("label_normalization", ['maxmin', 'maxminmax', 'std', 'None'])

    if args.label_normalization in {'None', 'maxminmax'}:   # large magnitude boost factors are not needed for these normalizations
        args.boost_factor = trial.suggest_float("boost_factor", 1.0, boost_factor_raw_max, log=True)  # 1.2 max for mixed geometric/Jaccard, 10 for Jaccard only
    else:  # std, maxmin
        args.boost_factor = trial.suggest_float("boost_factor", 1.0, boost_factor_norm_max, log=True)

    HIT = 63  # this depends on the `hit` in the RNN config, but we don't have access to it here 
    MAX_MAX_INJ_RELEVANT = min(RETURN_TOP_e2e, HIT)
    args.max_inj_relevant = trial.suggest_int("max_inj_relevant", 3, MAX_MAX_INJ_RELEVANT, log=True)

    if args.label_normalization == 'None':
        args.label_normalization = None  # the actual expected value in the code is None, not 'None'

    # Other hyperparameters for training
    args.num_epochs = 25
    args.learning_rate = trial.suggest_loguniform('learning_rate', 5e-7, 5e-6)
    args.encoder_learning_rate = args.learning_rate
    args.warmup_steps = int(trial.suggest_discrete_uniform('warmup_steps', 1000, 20000, 1000))
    args.encoder_warmup_steps = args.warmup_steps
    args.final_lr_ratio = trial.suggest_uniform('final_lr_ratio', 0.1, 0.1)#, 0.999)

    # Set name of experiment
    args.experiment_name = exp_name_prefix + f'_rboost{round(args.boost_factor, 3)}_norm-{args.label_normalization}_top{args.max_inj_relevant}'
    args.experiment_name += f'_lr{args.learning_rate:.2e}_warmup{args.warmup_steps}_finlrr{args.final_lr_ratio:.2f}'
    args.experiment_name += exp_name_suffix

    config = main.setup(args)  # Setup main training session
    best_values = main.main(config, trial)  # best metrics found during evaluation

    for name, value in best_values.items():
        trial.set_user_attr(name, value)  # log in database / study object

    return best_values[OPTIM_METRIC]


if __name__ == '__main__':

    utils.set_seed()  # randomize hyperparam search

    if task == 'rerank':
        study_name = 'recipNN_postprocess_reranking'  # This name is shared across jobs/processes
        objective = recipNN_rerank_objective
    elif task == 'smooth_labels':
        objective = smooth_labels_objective
    elif task == 'smooth_labels_e2e':
        objective = smooth_labels_e2e_objective

    storage_url = f'sqlite:////gpfs/data/ceickhof/gzerveas/RecipNN/recipNN_{task}_{dataset}_optuna.db'
    storage = optuna.storages.RDBStorage(url=storage_url, engine_kwargs={"connect_args": {"timeout": 100}})

    n_trials = 200
    sampler = TPESampler()  # TPESampler(**TPESampler.hyperopt_parameters())
    direction = 'minimize' if OPTIM_METRIC in NEG_METRICS else 'maximize'
    study_name += '_lr_warmup_hit'  # '_HEADnTORSO_hit'  #'_TRUE_learning_rate'
    study_name += '_' + exp_name_suffix
    study_name += f'_{direction}'  # append metric optimization direction to study name

    # random wait time in seconds to avoid writing to DB exactly at the same time
    wait_time = random.randint(0, 60)
    time.sleep(wait_time)

    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction=direction,
                                sampler=sampler,
                                pruner=MedianPruner(n_min_trials=10, n_startup_trials=5, n_warmup_steps=25000, interval_steps=5000))

    # fixed_params = {"y": best_params["y"]}
    # partial_sampler = optuna.samplers.PartialFixedSampler(fixed_params, study.sampler)
    # study.sampler = partial_sampler

    trials_df = study.trials_dataframe()  #(attrs=('number', 'value', 'params', 'state'))
    print(trials_df)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)  # last argument does garbage collection to avoid memory leak
    print("Hyperparameter optimization session complete!")
    trials_df = study.trials_dataframe()
    print(trials_df.describe())
    print("Top trials:")
    print(trials_df.sort_values(by='value', ascending=False).head(10))

    print("\nBest trial:\n{}".format(study.best_trial))
    print("All done!")
