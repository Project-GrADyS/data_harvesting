import mlflow
import argparse
from hyperopt import hp, fmin, tpe
from train import train

argparse = argparse.ArgumentParser()
argparse.add_argument("-E", type=str, required=True, help="MLflow experiment ID", dest="experiment_id")
args = argparse.parse_args()

if __name__ == "__main__":
    import yaml

    with open("params.yaml", "r") as f:
        config: dict = yaml.safe_load(f)
    
    mlflow.set_experiment(args.experiment_id)

    space = {
        "optimization": {
            "num_optimizer_steps": hp.choice("num_optimizer_steps", [1, 5, 10]),
            "lr": hp.loguniform("lr", -10, -3),
            "grad_clip": hp.choice("grad_clip", [0, 0.5, 1.0, 5.0]),
        }
    }

    def tune(tune_args: dict):
        run_config = config.copy()
        
        # tune_args contains overrides that should be merged into the base config
        for section, params in tune_args.items():
            if section not in run_config:
                run_config[section] = {}
            for key, value in params.items():
                run_config[section][key] = value

        lr = run_config['optimization']['lr']
        num_optimizer_steps = run_config['optimization']['num_optimizer_steps']
        grad_clip = run_config['optimization']['grad_clip']
        return train(run_config, 
                     run_name=f"lr_{lr}_steps_{num_optimizer_steps}_clip_{grad_clip}")

    best = fmin(
        fn=tune,
        space=space,
        algo=tpe.suggest,
        max_evals=10,
        show_progressbar=False
    )
    print("Best hyperparameters found:")
    print(best)