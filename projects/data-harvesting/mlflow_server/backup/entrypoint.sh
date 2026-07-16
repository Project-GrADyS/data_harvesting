#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

: "${BACKUP_CRON:=0 3 * * *}"
: "${TZ:=America/Sao_Paulo}"

if [ -f "/usr/share/zoneinfo/${TZ}" ]; then
    ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime
    echo "${TZ}" > /etc/timezone
else
    echo "Warning: timezone ${TZ} was not found; cron will use the container default timezone." >&2
fi

write_env_var() {
    local name="$1"
    printf 'export %s=%q\n' "$name" "${!name-}"
}

{
    write_env_var TZ
    write_env_var BACKUP_RETENTION_DAYS
    write_env_var BACKUP_REMOTE
    write_env_var BACKUP_REMOTE_PATH
    write_env_var RCLONE_CONFIG
    write_env_var POSTGRES_HOST
    write_env_var POSTGRES_USER
    write_env_var POSTGRES_PASSWORD
    write_env_var POSTGRES_DB
    write_env_var MINIO_ENDPOINT
    write_env_var MINIO_ROOT_USER
    write_env_var MINIO_ROOT_PASSWORD
    write_env_var MLFLOW_ARTIFACT_BUCKET
} > /etc/mlflow-backup.env

cat > /etc/cron.d/mlflow-backup <<EOF
SHELL=/bin/bash
CRON_TZ=${TZ}
${BACKUP_CRON} root set -a; source /etc/mlflow-backup.env; set +a; /backup/backup.sh >> /proc/1/fd/1 2>> /proc/1/fd/2
EOF

chmod 0644 /etc/cron.d/mlflow-backup

echo "Scheduled MLflow backup with cron expression: ${BACKUP_CRON}"
exec cron -f
