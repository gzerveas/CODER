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
OPTIM_METRIC = 'MRR@10' # 'MRR', 'nDCG@10'
NEG_METRICS = ['loss']

## Hyperparameter optimization settings ##

task = 'smooth_labels_e2e'  # 'rerank', 'smooth_labels', 'smooth_labels_e2e'
dataset = 'TripClick'  # 'TripClick', 'MSMARCO'
sim_type = 'mixed'  # SET HERE    #trial.suggest_categorical("sim_type", ['jaccard', 'geometric', 'mixed'])

# Different settings depending on target score type

if sim_type == 'jaccard':
    smoothing_study_name = 'recipNN_smooth_labels_JaccOnly_study'
    exp_name_prefix = 'AUTO_JaccOnly'
    boost_factor_raw_max = 10
    boost_factor_norm_max = 10
elif sim_type == 'geometric':
    smoothing_study_name = 'recipNN_smooth_labels_GeomOnly_study'
    exp_name_prefix = 'AUTO_GeomOnly'
    boost_factor_raw_max = 1.1
    boost_factor_norm_max = 3
else:  # mixed
    smoothing_study_name = 'recipNN_smooth_labels_study'
    exp_name_prefix = 'AUTO_RNN-'
    if task == 'smooth_labels':  # here we need to specify the RNN config; in e2e it is dynamically generated
        if dataset == 'TripClick':
            exp_name_prefix += 'l3b'  # random ID of RNN configuration
        else:
            exp_name_prefix += 'r67'
    boost_factor_raw_max = 1.2
    boost_factor_norm_max = 5

if dataset == 'MSMARCO':
    exp_name_suffix = '_CODER-TASB-IZc'
    if task == 'smooth_labels_e2e':
        train_config_file = '/gpfs/data/ceickhof/gzerveas/RecipNN/smooth_labels/configs/boost.config'  # path to config file for training
else:
    if task == 'smooth_labels':
        exp_name_suffix = '_TripClick_RAW_inc0.1_rel1_trainALL_valHEAD_CODER_repbert295k_xAk'
    elif task == 'smooth_labels_e2e':
        exp_name_suffix = '_TripClick_RAW_inc0.1_rel1_trainHEAD_valHEAD_CODER_repbert295k_xAk'
        train_config_file = '/users/gzerveas/data/gzerveas/RecipNN/smooth_labels/configs/TripClick/AUTOe2e_TripClick_CODER_RepBERT_xAk.config'  # path to config file for training

def recipNN_rerank_objective(trial):
    random.seed()  # re-randomize seed (it's fixed inside `main`), because it's needed for dir name suffix
    args = reciprocal_compute.run_parse_args()  # `argsparse` object
    config = utils.load_config(args)  # config dictionary, which potentially comes from a JSON file or command-line overrides
    args = utils.dict2obj(config)  # convert back to args object
    args.config_filepath = None  # the contents of a JSON file (if specified) have been loaded already, so prevent the `setup` from overwriting the Optuna overrides

    ## Optuna overrides
    args.sim_mixing_coef = trial.suggest_float('sim_mixing_coef', 1e-3, 1, log=True)
    args.k = trial.suggest_int('k', 3, 30)
    args.trust_factor = trial.suggest_categorical("trust_factor", [0, 0.5])
    args.k_exp = trial.suggest_int('k_exp', 1, 10)
    args.normalize = trial.suggest_categorical("normalize", ['max', 'mean', 'None'])
    args.weight_func = trial.suggest_categorical("weight_func", ['exp', 'linear'])
    if args.weight_func == 'exp':
        if args.normalize == 'None':  # NOTE: with arbitrary score values, the exponential parameter can easily lead to Inf!
            args.weight_func_param = trial.suggest_float("weight_func_param", 0.001, 1, log=True)
        else:
            args.weight_func_param = trial.suggest_float("weight_func_param", 1.0, 10.0, log=True)

    args = reciprocal_compute.setup(args)  # dump config file, create output dir, etc.

    best_values = recip_NN_rerank(args)  # run main function and get best metrics found during evaluation

    for name, value in best_values.items():
        trial.set_user_attr(name, value)  # log in database / study object

    return best_values[OPTIM_METRIC]


def smooth_labels_e2e_objective(trial):
    """Objective function for end-to-end hyperparameter optimization of recipNN label smooothing and subsequent training."""
    random.seed()  # re-randomize seed (it's fixed inside `main`), because it's needed for dir name suffix
    # Setup label smoothing
    args_rNN = reciprocal_compute.run_parse_args()  # parse arguments with `reciprocal_compute.py` parsing function
    config = utils.load_config(args_rNN)  # config dictionary, which potentially comes from a JSON file or command-line overrides
    args_rNN = utils.dict2obj(config)  # convert back to args object
    args_rNN.config_filepath = None  # the contents of a JSON file (if specified) have been loaded already, so prevent the `setup` from overwriting the Optuna overrides
    
    args_rNN.write_to_file = 'pickle'  # this is how we pass the target scores to `main` function

    # ReciprocalNN hyperparams
    args_rNN.sim_mixing_coef = trial.suggest_float('sim_mixing_coef', 1e-3, 1, log=True)
    args_rNN.k = trial.suggest_int('k', 3, 30)
    args_rNN.trust_factor = trial.suggest_categorical("trust_factor", [0, 0.5])
    args_rNN.k_exp = trial.suggest_int('k_exp', 1, 10)
    args_rNN.normalize = trial.suggest_categorical("normalize", ['max', 'mean', 'None'])
    args_rNN.weight_func = trial.suggest_categorical("weight_func", ['exp', 'linear'])
    if args_rNN.weight_func == 'exp':
        if args_rNN.normalize == 'None':  # NOTE: with arbitrary score values, the exponential parameter can easily lead to Inf!
            args_rNN.weight_func_param = trial.suggest_float("weight_func_param", 0.001, 1, log=True)
        else:
            args_rNN.weight_func_param = trial.suggest_float("weight_func_param", 1.0, 10.0, log=True)

    # ReciprocalNN label smoothing hyperparams
    args_rNN.rel_aggregation = trial.suggest_categorical("rel_aggregation", ['max', 'mean'])
    # args_rNN.redistribute = trial.suggest_categorical("redistribute", ['radically'])
    # args_rNN.norm_relevances = trial.suggest_categorical("norm_relevances", ['max', 'minmax', 'maxminmax', 'None', 'std'])
    # args_rNN.boost_factor = trial.suggest_float("boost_factor", 1.00, 10.0, log=False)
    ## args_rNN.redistr_prt = trial.suggest_float("redistr_prt", 0.01, 0.8, log=False)

    args_rNN = reciprocal_compute.setup(args_rNN)  # dump config file, create output dir, etc.

    # Run function to recompute relevance labels and write to disk as a file
    _ = extend_relevances(args_rNN)

    # Read configuration for training
    # parse arguments with `options.py` parsing function, to ensure all arguments expected by `load_config` are present
    args = options.run_parse_args('--config {}'.format(train_config_file).split(' '))

    config = utils.load_config(args)  # config dictionary, which comes from `train_config_file` JSON file
    args = utils.dict2obj(config)  # convert back to args object
    args.target_scores_path = args_rNN.out_filepath  # the file used to store smoothened labels becomes the target labels file for training
    args.config_filepath = None  # the contents of the JSON file have been loaded already, so prevent the `main.setup` from overwriting the Optuna overrides

    args.output_dir = os.path.join(args.output_dir, "AUTOe2e/")  # append to output dir to distinguish from other runs
    args.num_epochs = 40

    # Training Hyperparameters
    args.boost_relevant = "constant"
    args.label_normalization = trial.suggest_categorical("label_normalization", ['maxmin', 'maxminmax', 'std', 'None'])

    if args.label_normalization in {'None', 'maxminmax'}:   # large magnitude boost factors are not needed for these normalizations
        args.boost_factor = trial.suggest_float("boost_factor", 1.0, boost_factor_raw_max, log=True)  # 1.2 max for mixed geometric/Jaccard, 10 for Jaccard only
    else:
        args.boost_factor = trial.suggest_float("boost_factor", 1.0, boost_factor_norm_max, log=True)

    args.max_inj_relevant = trial.suggest_int("max_inj_relevant", 3, 300, log=True)

    if args.label_normalization == 'None':
        args.label_normalization = None  # the actual expected value in the code is None, not 'None'

    # Set name of experiment
    exp_name_pref = exp_name_prefix + args_rNN.random_suffix  # random ID of RNN configuration
    args.experiment_name = exp_name_pref + f'_rboost{round(args.boost_factor, 3)}_norm-{args.label_normalization}_top{args.max_inj_relevant}' + exp_name_suffix

    # Run main function for training
    config = main.setup(args)  # Setup main training session (logging, output dir, dump config file, etc.)
    best_values = main.main(config, trial)  # best metrics found during evaluation

    for name, value in best_values.items():
        trial.set_user_attr(name, value)  # log in database / study object

    return best_values[OPTIM_METRIC]


def smooth_labels_objective(trial):
    """Objective function for hyperparameter optimization of training with smooth labels"""
    random.seed()  # re-randomize seed (it's fixed inside `main`), because it's needed for dir name suffix

    # Run main function hyperparam optimization for training
    args = options.run_parse_args()  # `argsparse` object for training

    # Optuna overrides for main training
    config = utils.load_config(args)  # config dictionary, which potentially comes from a JSON file
    args = utils.dict2obj(config)  # convert back to args object
    args.config_filepath = None  # the contents of a JSON file (if specified) have been loaded already, so prevent the `main.setup` from overwriting the Optuna overrides

    args.num_epochs = 40

    # Hyperparameters
    args.boost_relevant = "constant"
    args.label_normalization = trial.suggest_categorical("label_normalization", ['None']) #['maxmin', 'maxminmax', 'std', 'None']) # TODO: RESTORE!

    if args.label_normalization in {'None', 'maxminmax'}:   # large magnitude boost factors are not needed for these normalizations
        args.boost_factor = trial.suggest_float("boost_factor", 1.0, boost_factor_raw_max, log=True)  # 1.2 max for mixed geometric/Jaccard, 10 for Jaccard only
    else:
        args.boost_factor = trial.suggest_float("boost_factor", 1.0, boost_factor_norm_max, log=True)

    args.max_inj_relevant = trial.suggest_int("max_inj_relevant", 3, 100, log=True)

    if args.label_normalization == 'None':
        args.label_normalization = None  # the actual expected value in the code is None, not 'None'

    # Set name of experiment
    args.experiment_name = exp_name_prefix + f'_rboost{round(args.boost_factor, 3)}_norm-{args.label_normalization}_top{args.max_inj_relevant}' + exp_name_suffix

    config = main.setup(args)  # Setup main training session
    best_values = main.main(config, trial)  # best metrics found during evaluation

    for name, value in best_values.items():
        trial.set_user_attr(name, value)  # log in database / study object

    return best_values[OPTIM_METRIC]


if __name__ == '__main__':

    utils.set_seed()  # randomize hyperparam search

    wait_time = random.randint(0, 30)  # random wait time in seconds to avoid writing to DB exactly at the same time
    time.sleep(wait_time)

    if task == 'rerank':
        study_name = 'recipNN_postprocess_reranking_study_normalization'  # This name is shared across jobs/processes
        objective = recipNN_rerank_objective
    elif task == 'smooth_labels':
        study_name = smoothing_study_name  # This name is shared across jobs/processes
        objective = smooth_labels_objective
    elif task == 'smooth_labels_e2e':
        study_name = 'recipNN_smooth_labels_e2e_study'  # This name is shared across jobs/processes
        objective = smooth_labels_e2e_objective

    storage = f'sqlite:////gpfs/data/ceickhof/gzerveas/RecipNN/recipNN_{task}_{dataset}_optuna.db'

    n_trials = 200
    sampler = TPESampler()  # TPESampler(**TPESampler.hyperopt_parameters())
    direction = 'minimize' if OPTIM_METRIC in NEG_METRICS else 'maximize'
    study_name += f'_{direction}'  # append metric optimization direction to study name

    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction=direction,
                                sampler=sampler,
                                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=80000, interval_steps=1))

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
