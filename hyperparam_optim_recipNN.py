import random

import optuna
from optuna.samplers import TPESampler, GridSampler
from optuna.pruners import MedianPruner

from options import *
from reciprocal_compute import rerank_reciprocal_neighbors, run_parse_args, setup, NORMALIZATION, WEIGHT_FUNC, WEIGHT_FUNC_PARAM
import utils

# Metric for hyperparam optimization.
# Can be different from "key_metric" of main, which determines the set of "best_values" and saved checkpoints
OPTIM_METRIC = 'NDCG@10'
NEG_METRICS = []


def objective(trial):
    random.seed()  # re-randomize seed (it's fixed inside `main`), because it's needed for dir name suffix
    args = run_parse_args()  # `argsparse` object
    args = setup(args)
    

    ## Optuna overrides
    args.sim_mixing_coef = trial.suggest_float('sim_mixing_coef', 1e-3, 1, log=True)
    args.k = trial.suggest_int('k', 3, 30)
    args.trust_factor = trial.suggest_categorical("trust_factor", [0, 0.5])
    args.k_exp = trial.suggest_int('k_exp', 1, 10) 
    args.normalize = trial.suggest_categorical("normalize", ['max', 'mean', 'None']) #[NORMALIZATION])  # constant
    args.weight_func = trial.suggest_categorical("weight_func", ['exp', 'linear']) #[WEIGHT_FUNC])  # constant
    args.weight_func_param = trial.suggest_float("weight_func_param", 0.1, 10, log=True) #trial.suggest_categorical("weight_func_param", [WEIGHT_FUNC_PARAM])  # constant

    best_values = rerank_reciprocal_neighbors(args)  # best metrics found during evaluation

    for name, value in best_values.items():
        trial.set_user_attr(name, value)  # log in database / study object

    return best_values[OPTIM_METRIC]


if __name__ == '__main__':

    storage = 'sqlite:////gpfs/data/ceickhof/gzerveas/RecipNN/recipNN_MSMARCO_optuna.db'
    study_name = 'recipNN_postprocess_reranking_study_normalization'  # This name is shared across jobs/processes
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
