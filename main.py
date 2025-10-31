import mlflow
import argparse
from hyperopt import hp, fmin, tpe
from train import train

argparse = argparse.ArgumentParser()
argparse.add_argument("-E", type=str, required=False, help="MLflow experiment ID", dest="experiment_name")
args = argparse.parse_args()

mlflow.set_tracking_uri("file:./mlruns")

if __name__ == "__main__":
    import yaml

    with open("params.yaml", "r") as f:
        config: dict = yaml.safe_load(f)
    
    mlflow.set_experiment(args.experiment_name if args.experiment_name else "default")

    space = {
        "optimization": {
            "num_optimizer_steps": hp.choice("num_optimizer_steps", [1, 5, 10, 20, 50]),
            "lr": hp.loguniform("lr", -10, -3),
            "grad_clip": hp.choice("grad_clip", [0, 0.5, 1.0, 5.0]),
        }
    }

    def tune(tune_args: dict):
        run_config = config.copy()
        
        tuned_arg_leafs: tuple[str, any] = []

        # tune_args contains overrides that should be merged into the base config
        for section, params in tune_args.items():
            if section not in run_config:
                run_config[section] = {}
            for key, value in params.items():
                run_config[section][key] = value
                tuned_arg_leafs.append((f"{section}.{key}", value))

        run_name = " ".join([f"{k}={v}" for k, v in tuned_arg_leafs])

        return -train(run_config, 
                      run_name=run_name)

    best = fmin(
        fn=tune,
        space=space,
        algo=tpe.suggest,
        max_evals=30,
        show_progressbar=False
    )
    print("Best hyperparameters found:")
    print(best)