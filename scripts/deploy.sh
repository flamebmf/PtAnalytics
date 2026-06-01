# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
#!/bin/bash
# ============================================================
# deploy.sh — развёртывание cam-analyzer в Podman на RHEL 9/10
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_DIR}/.env"

POD_NAME="${POD_NAME:-cam-pod}"
APP_IMAGE="${APP_IMAGE:-cam-analyzer:latest}"
REGISTRY="${REGISTRY:-}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
WITH_BUILD="${WITH_BUILD:-yes}"
WITH_LOCAL_PG="${WITH_LOCAL_PG:-no}"
WITH_GPU="${WITH_GPU:-no}"
WITH_SYSTEMD="${WITH_SYSTEMD:-no}"
DATA_ROOT="${DATA_ROOT:-/srv/cam-analyzer}"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
warn() { echo "[$(date '+%H:%M:%S')] WARN: $*" >&2; }
fail() { echo "ERROR: $*" >&2; exit 1; }

# ------------------------------------------------------------------
# 0. Load .env
# ------------------------------------------------------------------
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

if [[ "$WITH_LOCAL_PG" == "yes" ]]; then
    DB_HOST="${DB_HOST:-localhost}"
    DB_PASSWORD="${DB_PASSWORD:-cam_pass_$(openssl rand -hex 6)}"
else
    : "${DB_HOST:?DB_HOST не задан}"
    : "${DB_PASSWORD:?DB_PASSWORD не задан}"
fi

DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-cam}"
DB_USER="${DB_USER:-cam}"

log "=== Deploying cam-analyzer ==="
log "Pod:       $POD_NAME"
log "Data root: $DATA_ROOT"
log "DB:        $DB_HOST:$DB_PORT/$DB_NAME (local=$WITH_LOCAL_PG)"
log "GPU:       $WITH_GPU"
log "Build:     $WITH_BUILD"
log "Systemd:   $WITH_SYSTEMD"

# ------------------------------------------------------------------
# 1. Prerequisites
# ------------------------------------------------------------------
command -v podman &>/dev/null || fail "Podman не установлен"

mkdir -p "${DATA_ROOT}/frames" "${DATA_ROOT}/models" \
         "${DATA_ROOT}/pgdata" "${DATA_ROOT}/config" "${DATA_ROOT}/logs"

for f in settings.yaml cameras.yaml triggers.yaml; do
    if [[ -f "${PROJECT_DIR}/config/$f" ]]; then
        cp "${PROJECT_DIR}/config/$f" "${DATA_ROOT}/config/"
        log "Copied config/$f"
    fi
done

# Copy fine-tuned model if present
if [[ -f "${PROJECT_DIR}/training/fine-tuned.pt" ]]; then
    cp "${PROJECT_DIR}/training/fine-tuned.pt" "${DATA_ROOT}/models/"
    log "Copied fine-tuned.pt"
fi

chown -R 1001:0 "${DATA_ROOT}/config" "${DATA_ROOT}/logs" "${DATA_ROOT}/frames" "${DATA_ROOT}/models"
chmod -R 775 "${DATA_ROOT}/config" "${DATA_ROOT}/logs" "${DATA_ROOT}/frames" "${DATA_ROOT}/models"

# ------------------------------------------------------------------
# 2. Network
# ------------------------------------------------------------------
NETWORK_NAME="cam-net"
if ! podman network exists "$NETWORK_NAME" 2>/dev/null; then
    podman network create "$NETWORK_NAME"
    log "Network '$NETWORK_NAME' created"
fi

# ------------------------------------------------------------------
# 3. Build/pull image FIRST (before killing old pod)
# ------------------------------------------------------------------
if [[ "$WITH_BUILD" == "yes" ]]; then
    log "Building cam-analyzer image..."
    podman build -t "$APP_IMAGE" -f "${PROJECT_DIR}/docker/Containerfile.debian" "$PROJECT_DIR"
    log "Build complete"
elif [[ -n "$REGISTRY" ]]; then
    APP_IMAGE="${REGISTRY}/cam-analyzer:${IMAGE_TAG}"
    log "Pulling image: $APP_IMAGE"
    podman pull "$APP_IMAGE"
    log "Pull complete"
fi

# ------------------------------------------------------------------
# 4. Stop old pod (only after new image is ready)
# ------------------------------------------------------------------
if podman pod exists "$POD_NAME" 2>/dev/null; then
    log "Stopping old pod '$POD_NAME'..."
    podman pod stop "$POD_NAME" 2>/dev/null || true
    podman pod rm -f "$POD_NAME" 2>/dev/null || true
fi

# ------------------------------------------------------------------
# 5. Create pod
# ------------------------------------------------------------------
POD_CREATE_ARGS=(
    --name "$POD_NAME"
    --hostname cam-analyzer
    --network "$NETWORK_NAME"
    --publish "${HEALTH_PORT:-8090}:${HEALTH_PORT:-8090}"
)

podman pod create "${POD_CREATE_ARGS[@]}"
log "Pod '$POD_NAME' created"

# ------------------------------------------------------------------
# 7. PostgreSQL (optional)
# ------------------------------------------------------------------
if [[ "$WITH_LOCAL_PG" == "yes" ]]; then
    log "Starting PostgreSQL..."
    PG_IMAGE="${PG_IMAGE:-docker.io/pgvector/pgvector:pg17}"
    PG_CONTAINER="${PG_CONTAINER:-cam-postgres}"

    mkdir -p "${DATA_ROOT}/pgdata"

    INIT_DIR="${PROJECT_DIR}/data/pg-init"
    mkdir -p "$INIT_DIR"

    cat > "$INIT_DIR/01-init.sql" <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
SQL

    podman run -d --pod "$POD_NAME" --name "$PG_CONTAINER" --restart unless-stopped \
        -e POSTGRES_DB="$DB_NAME" \
        -e POSTGRES_USER="$DB_USER" \
        -e POSTGRES_PASSWORD="$DB_PASSWORD" \
        -e POSTGRES_HOST_AUTH_METHOD=md5 \
        -v "${DATA_ROOT}/pgdata:/var/lib/postgresql/data:Z" \
        -v "${INIT_DIR}/01-init.sql:/docker-entrypoint-initdb.d/01-init.sql:Z" \
        --health-cmd "pg_isready -U ${DB_USER} -d ${DB_NAME}" \
        --health-interval 5s --health-timeout 3s --health-retries 10 \
        --health-start-period 15s \
        "$PG_IMAGE"

    log "Waiting for PostgreSQL..."
    for i in $(seq 1 30); do
        if podman exec "$PG_CONTAINER" pg_isready -U "$DB_USER" -d "$DB_NAME" &>/dev/null; then
            log "PostgreSQL ready"
            break
        fi
        if [[ $i -eq 30 ]]; then
            podman logs "$PG_CONTAINER" 2>&1 | tail -20
            fail "PostgreSQL failed to start"
        fi
        sleep 2
    done
    DB_HOST="localhost"
fi

# ------------------------------------------------------------------
# 8. Start cam-analyzer
# ------------------------------------------------------------------
APP_RUN_ARGS=(
    -d
    --pod "$POD_NAME"
    --name cam-analyzer
    --restart unless-stopped
    -e TZ="${TZ:-Europe/Moscow}"
    -e DB_HOST="$DB_HOST"
    -e DB_PORT="$DB_PORT"
    -e DB_NAME="$DB_NAME"
    -e DB_USER="$DB_USER"
    -e DB_PASSWORD="$DB_PASSWORD"
    -e HEALTH_PORT="${HEALTH_PORT:-8090}"
    -e YOLO_CONFIG_DIR="/app/models/ultralytics"
    -e PADDLE_HOME="/app/models/paddle"
    -e PADDLEOCR_HOME="/app/models/paddleocr"
    -e PADDLEX_HOME="/app/models/paddlex"
    -e XDG_CACHE_HOME="/app/models/cache"
    -e HOME="/app/models/home"
    -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
    -e MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
    -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True"
    -v "${DATA_ROOT}/frames:/data/frames:Z"
    -v "${DATA_ROOT}/models:/app/models:Z"
    -v "${DATA_ROOT}/config:/app/config:Z"
    -v "${DATA_ROOT}/logs:/app/logs:Z"
)

if [[ "$WITH_GPU" == "yes" ]]; then
    APP_RUN_ARGS+=(--device nvidia.com/gpu=all --security-opt=label=disable)
fi

podman run "${APP_RUN_ARGS[@]}" \
    --health-cmd "python3 -c \"import urllib.request; r=urllib.request.urlopen('http://localhost:8090/health', timeout=3); import json; d=json.load(r); exit(0 if d.get('status') in ('healthy','degraded') else 1)\"" \
    --health-interval 15s --health-timeout 5s --health-retries 3 \
    --health-start-period 30s \
    "$APP_IMAGE"
log "cam-analyzer started"

# ------------------------------------------------------------------
# 9. Systemd (optional)
# ------------------------------------------------------------------
if [[ "$WITH_SYSTEMD" == "yes" ]]; then
    log "Generating systemd unit..."
    mkdir -p "${HOME}/.config/systemd/user"

    podman generate systemd --new --name --files "$POD_NAME"
    for f in pod-*.service; do
        mv "$f" "${HOME}/.config/systemd/user/"
    done

    if systemctl --user daemon-reload 2>/dev/null; then
        systemctl --user enable podman-pod-${POD_NAME}.service 2>/dev/null || true
        log "systemd units installed"
        log "Для автозапуска: loginctl enable-linger $USER"
    else
        warn "systemd-user not available, units saved to ~/.config/systemd/user/"
    fi
fi

# ------------------------------------------------------------------
# 10. Health check
# ------------------------------------------------------------------
sleep 3
if podman pod ps --filter "name=$POD_NAME" --format "{{.Status}}" | grep -q "Running"; then
    log "=== Pod '$POD_NAME' running ==="
    podman pod ps --filter "name=$POD_NAME"
    echo ""
    log "Logs:  podman logs cam-analyzer"
    log "Shell: podman exec -it cam-analyzer /bin/bash"
else
    log "=== Pod status ==="
    podman pod ps --filter "name=$POD_NAME"
    podman logs cam-analyzer 2>&1 | tail -30
    fail "Pod failed to start"
fi
