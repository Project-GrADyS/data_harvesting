import mlflow
import argparse
from data_harvesting.train import train

parser = argparse.ArgumentParser()
parser.add_argument("-E", type=str, required=False, help="MLflow experiment ID", dest="experiment_name")

run_group = parser.add_mutually_exclusive_group()
run_group.add_argument("-R", type=str, required=False, help="MLflow run name", dest="run_name")
run_group.add_argument("--resume-run-id", type=str, required=False, default=None, help="MLflow run ID to resume logging into")

parser.add_argument("--resume-checkpoint", type=str, required=False, default=None, help="Checkpoint artifact file name to resume from (e.g. checkpoint_step_100000.pt)")
args = parser.parse_args()

if args.resume_run_id and not args.resume_checkpoint:
    parser.error("--resume-checkpoint is required when --resume-run-id is set")
if args.resume_checkpoint and not args.resume_run_id:
    parser.error("--resume-run-id is required when --resume-checkpoint is set")

mlflow.set_tracking_uri("file:./mlruns")

if __name__ == "__main__":
    import yaml

    with open("params.yaml", "r") as f:
        config: dict = yaml.safe_load(f)
    
    mlflow.set_experiment(args.experiment_name if args.experiment_name else "default")

    train(
        config,
        run_name=args.run_name,
        resume_checkpoint=args.resume_checkpoint,
        resume_run_id=args.resume_run_id,
    )