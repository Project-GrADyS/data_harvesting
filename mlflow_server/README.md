# MLflow Server

This directory runs a private MLflow tracking server with Postgres for metadata
and MinIO for artifact storage.

## First-time setup

```bash
cd mlflow_server
cp .env.example .env
```

Edit `.env` and replace the Postgres and MinIO passwords. Keep
`MLFLOW_HOST=127.0.0.1` for SSH-tunnel-only access, or set it to the VPS VPN
interface IP if you want other machines on the private network to connect.

Start the stack:

```bash
docker compose up -d --build
```

Open MLflow at:

```text
http://<vpn-host>:5000
```

The MinIO console is exposed only on the configured private bind address:

```text
http://<vpn-host>:9001
```

## Client usage

For training and evaluation from this repository, point MLflow clients at the
server:

```bash
export MLFLOW_TRACKING_URI=http://<vpn-host>:5000
uv run python main.py -E default
uv run python evaluate.py --tracking-uri "$MLFLOW_TRACKING_URI" --run-id <MLFLOW_RUN_ID> --num-runs 10
```

You can also pass the tracking URI directly:

```bash
uv run python main.py --tracking-uri http://<vpn-host>:5000 -E default
uv run python tune.py --tracking-uri http://<vpn-host>:5000 -E tuning
```

MLflow proxies artifact uploads to MinIO, so clients only need access to the
MLflow server URL. They do not need MinIO credentials or direct MinIO network
access.

## Backups

The Compose stack includes a `backup` service that backs up both durable stores:

- Postgres MLflow metadata via `pg_dump -Fc`.
- MinIO artifacts via rclone's S3 backend.

Backups are uploaded to OneDrive with rclone. By default they run daily at
03:00 São Paulo time and keep 30 days of Postgres dumps and artifact manifests.
Artifact objects are copied into a rolling `artifacts/current/` directory.

First configure rclone on the VPS:

```bash
mkdir -p rclone
docker run --rm -it \
  -v "$PWD/rclone:/config/rclone" \
  rclone/rclone config
```

Create a OneDrive remote named `onedrive`. The rclone config file under
`mlflow_server/rclone/` contains credentials and is ignored by git. The backup
container mounts this directory read-write because rclone needs to persist
refreshed OneDrive tokens.

Start the scheduler with the rest of the stack:

```bash
docker compose up -d --build
```

Run a one-off backup:

```bash
docker compose run --rm backup /backup/backup.sh
```

Run a dry-run backup that verifies local dump/listing behavior and rclone
operations without changing OneDrive:

```bash
docker compose run --rm backup /backup/backup.sh --dry-run
```

Restore instructions are in `backup/RESTORE.md`.

## Security notes

This Compose stack is intended for private/VPN access. Do not expose MLflow or
MinIO directly to the public internet without adding TLS and authentication at a
reverse proxy layer.

Existing file-backed runs under `mlruns/` are left as legacy data. New work
should use `MLFLOW_TRACKING_URI` or `--tracking-uri` to log to this server.
