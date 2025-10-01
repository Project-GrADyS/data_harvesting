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
    
    if args.experiment_name:
        mlflow.set_experiment(args.experiment_name)
    train(config)