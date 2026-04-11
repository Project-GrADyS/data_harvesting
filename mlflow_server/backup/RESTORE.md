# Restore From OneDrive Backups

These commands assume `mlflow_server/.env` is configured, the rclone OneDrive
remote is available at `mlflow_server/rclone/rclone.conf`, and the Compose stack
is running.

List available Postgres dumps:

```bash
docker compose run --rm backup bash -lc '
  rclone --config /config/rclone/rclone.conf lsf "${BACKUP_REMOTE}:${BACKUP_REMOTE_PATH}/postgres"
'
```

Pick a dump timestamp, then stop MLflow writes:

```bash
docker compose stop mlflow
```

Download and restore Postgres:

```bash
dump_name=<TIMESTAMP>.dump
docker compose run --rm backup bash -lc '
  set -euo pipefail
  rclone --config /config/rclone/rclone.conf copyto \
    "${BACKUP_REMOTE}:${BACKUP_REMOTE_PATH}/postgres/'"${dump_name}"'" \
    "/work/'"${dump_name}"'"
  PGPASSWORD="${POSTGRES_PASSWORD}" pg_restore \
    -h postgres \
    -U "${POSTGRES_USER}" \
    -d "${POSTGRES_DB}" \
    --clean \
    --if-exists \
    "/work/'"${dump_name}"'"
'
```

Restore MinIO artifacts from OneDrive:

```bash
docker compose run --rm backup bash -lc '
  set -euo pipefail
  export RCLONE_CONFIG_MLFLOW_MINIO_TYPE=s3
  export RCLONE_CONFIG_MLFLOW_MINIO_PROVIDER=Minio
  export RCLONE_CONFIG_MLFLOW_MINIO_ACCESS_KEY_ID="${MINIO_ROOT_USER}"
  export RCLONE_CONFIG_MLFLOW_MINIO_SECRET_ACCESS_KEY="${MINIO_ROOT_PASSWORD}"
  export RCLONE_CONFIG_MLFLOW_MINIO_ENDPOINT="${MINIO_ENDPOINT}"
  rclone --config /config/rclone/rclone.conf copy \
    "${BACKUP_REMOTE}:${BACKUP_REMOTE_PATH}/artifacts/current" \
    "mlflow_minio:${MLFLOW_ARTIFACT_BUCKET}"
'
```

Restart MLflow:

```bash
docker compose up -d mlflow
```

Validate by opening the MLflow UI and loading a known run/model.
