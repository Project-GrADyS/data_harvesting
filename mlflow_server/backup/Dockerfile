FROM postgres:16

ARG TARGETARCH=amd64

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        cron \
        curl \
        unzip \
        tzdata \
    && curl -fsSL https://rclone.org/install.sh | bash \
    && curl -fsSL "https://dl.min.io/client/mc/release/linux-${TARGETARCH}/mc" -o /usr/local/bin/mc \
    && chmod +x /usr/local/bin/mc \
    && rm -rf /var/lib/apt/lists/*

COPY backup.sh entrypoint.sh /backup/

RUN chmod +x /backup/backup.sh /backup/entrypoint.sh

WORKDIR /backup

ENTRYPOINT ["/backup/entrypoint.sh"]
