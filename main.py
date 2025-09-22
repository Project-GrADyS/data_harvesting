import mlflow
import argparse
from train import train

argparse = argparse.ArgumentParser()
argparse.add_argument("-E", type=str, required=True, help="MLflow experiment ID", dest="experiment_id")
args = argparse.parse_args()

if __name__ == "__main__":
    import yaml

    with open("params.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    mlflow.set_experiment(args.experiment_id or config.get("experiment_id", "default_experiment"))

    train(config)