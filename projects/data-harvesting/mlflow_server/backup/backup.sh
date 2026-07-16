#!/usr/bin/env bash
set -Eeuo pipefail

dry_run=0
if [ "${1:-}" = "--dry-run" ]; then
    dry_run=1
fi

: "${BACKUP_RETENTION_DAYS:=30}"
: "${BACKUP_REMOTE:=onedrive}"
: "${BACKUP_REMOTE_PATH:=mlflow-backups/data-harvesting}"
: "${RCLONE_CONFIG:=/config/rclone/rclone.conf}"
: "${POSTGRES_HOST:=postgres}"
: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${POSTGRES_PASSWORD:?POSTGRES_PASSWORD is required}"
: "${POSTGRES_DB:?POSTGRES_DB is required}"
: "${MINIO_ENDPOINT:=http://minio:9000}"
: "${MINIO_ROOT_USER:?MINIO_ROOT_USER is required}"
: "${MINIO_ROOT_PASSWORD:?MINIO_ROOT_PASSWORD is required}"
: "${MLFLOW_ARTIFACT_BUCKET:=mlflow-artifacts}"

if [ ! -f "${RCLONE_CONFIG}" ]; then
    echo "Missing rclone config at ${RCLONE_CONFIG}. Configure OneDrive and mount it at ./rclone/rclone.conf." >&2
    exit 2
fi

timestamp="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
workdir="/work/${timestamp}"
mkdir -p "${workdir}"

remote_root="${BACKUP_REMOTE}:${BACKUP_REMOTE_PATH}"
postgres_dump="${workdir}/postgres-${timestamp}.dump"
artifact_manifest="${workdir}/artifacts-${timestamp}.txt"

echo "Starting MLflow backup ${timestamp}"

export PGPASSWORD="${POSTGRES_PASSWORD}"
pg_dump \
    -h "${POSTGRES_HOST}" \
    -U "${POSTGRES_USER}" \
    -d "${POSTGRES_DB}" \
    -Fc \
    -f "${postgres_dump}"

mc alias set mlflow-minio "${MINIO_ENDPOINT}" "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}" >/dev/null
mc find "mlflow-minio/${MLFLOW_ARTIFACT_BUCKET}" > "${artifact_manifest}"

export RCLONE_CONFIG_MLFLOW_MINIO_TYPE=s3
export RCLONE_CONFIG_MLFLOW_MINIO_PROVIDER=Minio
export RCLONE_CONFIG_MLFLOW_MINIO_ACCESS_KEY_ID="${MINIO_ROOT_USER}"
export RCLONE_CONFIG_MLFLOW_MINIO_SECRET_ACCESS_KEY="${MINIO_ROOT_PASSWORD}"
export RCLONE_CONFIG_MLFLOW_MINIO_ENDPOINT="${MINIO_ENDPOINT}"

rclone_flags=(
    --config "${RCLONE_CONFIG}"
    --transfers 4
    --checkers 8
)

if [ "${dry_run}" -eq 1 ]; then
    rclone_flags+=(--dry-run)
    echo "Running in dry-run mode; no OneDrive changes will be made."
fi

rclone "${rclone_flags[@]}" mkdir "${remote_root}/postgres"
rclone "${rclone_flags[@]}" mkdir "${remote_root}/artifacts/manifests"
rclone "${rclone_flags[@]}" mkdir "${remote_root}/artifacts/current"

rclone "${rclone_flags[@]}" copyto \
    "${postgres_dump}" \
    "${remote_root}/postgres/${timestamp}.dump"

rclone "${rclone_flags[@]}" copyto \
    "${artifact_manifest}" \
    "${remote_root}/artifacts/manifests/${timestamp}.txt"

rclone "${rclone_flags[@]}" copy \
    --create-empty-src-dirs \
    "mlflow_minio:${MLFLOW_ARTIFACT_BUCKET}" \
    "${remote_root}/artifacts/current"

if [ "${dry_run}" -eq 1 ]; then
    echo "Skipping retention cleanup in dry-run mode."
else
    rclone "${rclone_flags[@]}" delete \
        "${remote_root}/postgres" \
        --min-age "${BACKUP_RETENTION_DAYS}d" \
        --include "*.dump"

    rclone "${rclone_flags[@]}" delete \
        "${remote_root}/artifacts/manifests" \
        --min-age "${BACKUP_RETENTION_DAYS}d" \
        --include "*.txt"
fi

rm -rf "${workdir}"

echo "Completed MLflow backup ${timestamp}"
