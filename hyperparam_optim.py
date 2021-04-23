import random

import optuna
from optuna.samplers import TPESampler

from options import *
from main import main, setup
import utils

# Metric for hyperparam optimization.
# Can be different from "key_metric" of main, which determines the set of "best_values" and saved checkpoints
OPTIM_METRIC = 'MRR'


def objective(trial):
    random.seed()  # re-randomize seed (it's fixed inside `main`), because it's needed for dir name suffix
    args = run_parse_args()  # `argsparse` object

    config = utils.load_config(args)  # config dictionary, which potentially comes from a JSON file
    args = utils.dict2obj(config)  # convert back to args object
    args.config_filepath = None  # the contents of a JSON file (if specified) have been loaded already, so prevent the `main.setup` from overwriting the Optuna overrides

    # Optuna overrides
    args.scoring_mode = trial.suggest_categorical('scoring_mode', ['raw', 'cross_attention', 'dot_product', 'cross_attention_gelu'])
    args.optimizer = trial.suggest_categorical('optimizer', ['AdamW', 'RAdam'])
    args.weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 5e-2)
    args.num_candidates = trial.suggest_int('num_candidates', 10, 400)
    args.num_inbatch_neg = trial.suggest_int('num_inbatch_neg', 10, 400)
    args.learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
    args.encoder_delay = int(trial.suggest_discrete_uniform('encoder_delay', 0, 20000, 5000))
    args.encoder_learning_rate = trial.suggest_uniform('encoder_learning_rate', 0.1*args.learning_rate, 2*args.learning_rate) #trial.suggest_loguniform('encoder_learning_rate', 1e-6, 1e-4)
    args.warmup_steps = int(trial.suggest_discrete_uniform('warmup_steps', 1000, 20000, 1000))
    args.encoder_warmup_steps = int(trial.suggest_discrete_uniform('encoder_warmup_steps', 5000, 20000, 1000))
    args.final_lr_ratio = trial.suggest_uniform('final_lr_ratio', 0.9999, 0.9999)
    args.adam_epsilon = trial.suggest_loguniform('adam_epsilon', 1e-8, 2e-6)

    args.num_layers = trial.suggest_int('num_layers', 1, 6)
    max_lognumheads = 4 if args.num_layers > 4 else 5
    args.num_heads = 2 ** trial.suggest_int('log2_num_heads', 3, max_lognumheads)
    max_dmodel = 768 if args.num_layers > 4 else 1024
    args.d_model = int(trial.suggest_discrete_uniform('d_model', 768//2, max_dmodel, args.num_heads))  # divisible by num_heads
    if args.d_model % args.num_heads != 0:
        raise ValueError("'d_model' must be divisible by 'num_heads'")
    max_dimFF = min(2*args.d_model, 1500)
    args.dim_feedforward = trial.suggest_int('dim_feedforward', int(0.5*args.d_model), max_dimFF)

    config = setup(args)  # configuration dictionary containing the arguments as specified in main.py and overriden above
    best_values = main(config)  # best metrics found during evaluation

    for name, value in best_values.items():
        trial.set_user_attr(name, value)  # log in database / study object

    return best_values[OPTIM_METRIC]


if __name__ == '__main__':

    storage = 'sqlite:////gpfs/data/ceickhof/dcohen_share/mdst_optuna.db'
    study_name = 'mdst_study_transp_Qenc_repbert'  # This name is shared across jobs
    n_trials = 10
    sampler = TPESampler()  # TPESampler(**TPESampler.hyperopt_parameters())
    direction = 'minimize' if OPTIM_METRIC in NEG_METRICS else 'maximize'

    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction=direction,
                                sampler=sampler)
    trials_df = study.trials_dataframe()  #(attrs=('number', 'value', 'params', 'state'))
    print(trials_df)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)  # last argument does garbage collection to avoid memory leak
    print("Hyperparameter optimization session complete!")
    trials_df = study.trials_dataframe()
    print(trials_df.describe())
    print("Top trials:")
    print(trials_df.sort_values(by='value', ascending=False).head(10))

    print("\nBest trial:\n{}".format(study.best_trial))
