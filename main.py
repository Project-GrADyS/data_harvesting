import mlflow
import argparse
from hyperopt import hp, fmin, tpe
from train import train

argparse = argparse.ArgumentParser()
argparse.add_argument("-E", type=str, required=True, help="MLflow experiment ID", dest="experiment_id")
args = argparse.parse_args()

mlflow.set_tracking_uri("file:./mlruns")

if __name__ == "__main__":
    import yaml

    with open("params.yaml", "r") as f:
        config: dict = yaml.safe_load(f)
    
    mlflow.set_experiment(args.experiment_id)

    space = {
        "training": {
            "batch_size": hp.choice("batch_size", [64, 256, 512, 1024, 2048]),
        },
        "collector": {
            "frames_per_batch": hp.choice("frames_per_batch", [256, 512, 1024, 2048, 4096])
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
        max_evals=20,
        show_progressbar=False
    )
    print("Best hyperparameters found:")
    print(best)