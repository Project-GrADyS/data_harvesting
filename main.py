import mlflow
import argparse
import os
from hyperopt import hp, fmin, tpe
from data_harvesting.train import train

argparse = argparse.ArgumentParser()
argparse.add_argument("-E", type=str, required=False, help="MLflow experiment ID", dest="experiment_name")
argparse.add_argument("-R", type=str, required=False, help="MLflow run ID", dest="run_id")
argparse.add_argument(
    "--tracking-uri",
    default=os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"),
    help="MLflow tracking URI (defaults to MLFLOW_TRACKING_URI or file:./mlruns)",
)
args = argparse.parse_args()

mlflow.set_tracking_uri(args.tracking_uri)

if __name__ == "__main__":
    import yaml

    with open("params.yaml", "r") as f:
        config: dict = yaml.safe_load(f)
    
    mlflow.set_experiment(args.experiment_name if args.experiment_name else "default")

    train(config, run_name=args.run_id)
