import optuna
from optuna.samplers import TPESampler
from main import main, run_parse_args, setup, NEG_METRICS, METRICS

# Metric for hyperparam optimization.
# Can be different from "key_metric" of main, which determines the set of "best_values" and saved checkpoints
OPTIM_METRIC = 'MRR'


def objective(trial):

    args = run_parse_args()  # `argsparse` object
    config = setup(args)  # configuration dictionary containing the defaults as specified in main.py

    # Optuna overrides
    config["optimizer"] = trial.suggest_categorical('optimizer', 'AdamW', 'RAdam')
    config["weight_decay"] = trial.suggest_loguniform('weight_decay', 1e-6, 5e-2)
    config["num_candidates"] = trial.suggest_int('num_candidates', 10, 400)
    config["num_inbatch_neg"] = trial.suggest_int('num_inbatch_neg', 10, 400)
    config["learning_rate"] = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
    config["encoder_learning_rate"] = trial.suggest_loguniform('encoder_learning_rate', 1e-6, 1e-4)
    config["warmup_steps"] = int(trial.suggest_discrete_uniform('warmup_steps', 1000, 20000, 1000))
    config["final_lr_ratio"] = trial.suggest_uniform('final_lr_ratio', 1e-1, 0.99)
    config["adam_epsilon"] = trial.suggest_loguniform('adam_epsilon', 1e-8, 2e-6)

    config["num_layers"] = trial.suggest_int('num_layers', 1, 6)
    config["num_heads"] = 2 ** trial.suggest_int('log2_num_heads', 3, 6)
    config["d_model"] = int(trial.suggest_discrete_uniform('d_model', 768//2, 1024, 8))  # divisible by num_heads
    config["dim_feedforward"] = trial.suggest_int('dim_feedforward', int(0.75*config["d_model"]), 2*config["d_model"])

    best_values = main(config)  # best metrics found during evaluation

    for name, value in best_values:
        trial.set_user_attr(name, value)  # log in database / study object

    return best_values[OPTIM_METRIC]



if __name__ == '__main__':

    storage = 'sqlite:///mdst_optuna.db'
    study_name = 'mdst_study_DEBUG'  # 'mdst_study_0'
    n_trials = 10
    sampler = TPESampler()  # TPESampler(**TPESampler.hyperopt_parameters())
    direction = 'minimize' if OPTIM_METRIC in NEG_METRICS else 'maximize'

    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction=direction,
                                sampler=sampler)
    trials_df = study.trials_dataframe()  #(attrs=('number', 'value', 'params', 'state'))
    print(trials_df)
    study.optimize(objective, n_trials=n_trials)
    print("Hyperparameter optimization session complete!")
    trials_df = study.trials_dataframe()
    print(trials_df)
    print("\nBest trial:\n{}".format(study.best_trial))
