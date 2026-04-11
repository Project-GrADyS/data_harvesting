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

Back up both named Docker volumes:

- `mlflow_server_postgres_data`: MLflow metadata.
- `mlflow_server_minio_data`: artifact objects and logged models.

For Postgres, prefer a logical dump:

```bash
docker compose exec postgres sh -c 'pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB"' > mlflow_postgres.sql
```

For MinIO, use your VPS backup tooling or an S3-compatible copy tool against the
`mlflow-artifacts` bucket.

## Security notes

This Compose stack is intended for private/VPN access. Do not expose MLflow or
MinIO directly to the public internet without adding TLS and authentication at a
reverse proxy layer.

Existing file-backed runs under `mlruns/` are left as legacy data. New work
should use `MLFLOW_TRACKING_URI` or `--tracking-uri` to log to this server.
