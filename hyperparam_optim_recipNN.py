import random

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
OPTIM_METRIC = 'NDCG@10'
NEG_METRICS = []


def recipNN_rerank_objective(trial):
    random.seed()  # re-randomize seed (it's fixed inside `main`), because it's needed for dir name suffix
    args = reciprocal_compute.run_parse_args()  # `argsparse` object
    args = reciprocal_compute.setup(args)
    

    ## Optuna overrides
    args.sim_mixing_coef = trial.suggest_float('sim_mixing_coef', 1e-3, 1, log=True)
    args.k = trial.suggest_int('k', 3, 30)
    args.trust_factor = trial.suggest_categorical("trust_factor", [0, 0.5])
    args.k_exp = trial.suggest_int('k_exp', 1, 10) 
    args.normalize = trial.suggest_categorical("normalize", ['max', 'mean', 'None']) #[NORMALIZATION])  # constant
    args.weight_func = trial.suggest_categorical("weight_func", ['exp', 'linear']) #[WEIGHT_FUNC])  # constant
    if args.weight_func == 'exp':
        args.weight_func_param = trial.suggest_float("weight_func_param", 0.001, 10, log=True)
        args.weight_func_param = trial.suggest_float("weight_func_param", 1.0, 1.0, log=True)  # constant

    best_values = recip_NN_rerank(args)  # run main function and get best metrics found during evaluation

    for name, value in best_values.items():
        trial.set_user_attr(name, value)  # log in database / study object

    return best_values[OPTIM_METRIC]


def smooth_labels_objective(trial):
    random.seed()  # re-randomize seed (it's fixed inside `main`), because it's needed for dir name suffix
    # Setup label smoothing
    args_rNN = reciprocal_compute.run_parse_args()  # `argsparse` object
    args_rNN = reciprocal_compute.setup(args_rNN)
    

    # Optuna overrides
    # args_rNN.sim_mixing_coef = trial.suggest_float('sim_mixing_coef', 1e-3, 1, log=True)
    
    args_rNN.rel_aggregation = trial.suggest_categorical("rel_aggregation", ['max', 'mean'])
    args_rNN.redistribute = trial.suggest_categorical("redistribute", ['fully', 'partially', 'radically'])
    args_rNN.norm_relevances = trial.suggest_categorical("norm_relevances", ['max', 'minmax', 'None', 'std'])
    args_rNN.boost_factor = trial.suggest_float("boost_factor", 1.00, 10.0, log=False)
    # args_rNN.redistr_prt = trial.suggest_float("redistr_prt", 0.01, 0.8, log=False)

    _ = extend_relevances(args_rNN, trial)  # run function to recompute relevance labels and store them as a file

    # Run main function (and optionally, hyperparam optimization) for training
    args = options.run_parse_args()  # `argsparse` object for training
    args.target_scores_path = args_rNN.out_filepath  # the file used to store smoothened labels becoms the input labels file for training
    
    # Optuna overrides for main training
    # config = utils.load_config(args)  # config dictionary, which potentially comes from a JSON file
    # args = utils.dict2obj(config)  # convert back to args object
    # args.config_filepath = None  # the contents of a JSON file (if specified) have been loaded already, so prevent the `main.setup` from overwriting the Optuna overrides
    # args.some_training_param = trial.suggest_float("some_training_param", 0.01, 0.8, log=False)

    config = main.setup(args)  # Setup main training session
    best_values = main.main(config, trial)  # best metrics found during evaluation

    for name, value in best_values.items():
        trial.set_user_attr(name, value)  # log in database / study object

    return best_values[OPTIM_METRIC]


if __name__ == '__main__':
    
    task = 'rerank'  # 'smooth_labels'
    if task == 'rerank':
        study_name = 'recipNN_postprocess_reranking_study_normalization'  # This name is shared across jobs/processes
        objective = recipNN_rerank_objective
    elif task == 'smooth_labels':
        study_name = 'recipNN_smooth_labels_study'  # This name is shared across jobs/processes
        objective = recipNN_rerank_objective

    storage = 'sqlite:////gpfs/data/ceickhof/gzerveas/RecipNN/recipNN_MSMARCO_optuna.db' #'sqlite:////gpfs/data/ceickhof/gzerveas/RecipNN/recipNN_TripClick_optuna.db'
    
    n_trials = 200
    sampler = TPESampler()  # TPESampler(**TPESampler.hyperopt_parameters())
    direction = 'minimize' if OPTIM_METRIC in NEG_METRICS else 'maximize'

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
