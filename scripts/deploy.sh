#!/bin/bash
# ============================================================
# deploy.sh — развёртывание cam-analyzer в Podman на RHEL 9/10
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_DIR}/.env"

# Конфигурация pod
POD_NAME="${POD_NAME:-cam-pod}"
APP_IMAGE="${APP_IMAGE:-cam-analyzer:latest}"
REGISTRY="${REGISTRY:-}"                        # docker.io/user  или  registry.example.com/project
IMAGE_TAG="${IMAGE_TAG:-latest}"                # тег образа в registry
WITH_BUILD="${WITH_BUILD:-yes}"                 # yes=собрать локально, no=взять из registry
MQTT_IMAGE="${MQTT_IMAGE:-docker.io/eclipse-mosquitto:2-openssl}"
WITH_MQTT="${WITH_MQTT:-no}"              # yes — запустить MQTT в поде
WITH_LOCAL_PG="${WITH_LOCAL_PG:-no}"      # yes — запустить PG в поде (игнорирует DB_HOST)
WITH_GPU="${WITH_GPU:-no}"                # yes — пробросить GPU
WITH_SYSTEMD="${WITH_SYSTEMD:-no}"        # yes — установить systemd сервис
WITH_TEST_STREAM="${WITH_TEST_STREAM:-no}" # yes — проверить камеры перед деплоем
DATA_ROOT="${DATA_ROOT:-/srv/cam-analyzer}"  # корень данных на хосте

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
warn() { echo "[$(date '+%H:%M:%S')] WARN: $*" >&2; }
fail() { echo "ERROR: $*" >&2; exit 1; }

# ------------------------------------------------------------------
# 0. Загрузка .env
# ------------------------------------------------------------------
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

# Если PG локальный — всё в поде, auto-generate пароль если не задан
if [[ "$WITH_LOCAL_PG" == "yes" ]]; then
    DB_HOST="${DB_HOST:-localhost}"
    DB_PASSWORD="${DB_PASSWORD:-cam_pass_$(openssl rand -hex 6)}"
else
    : "${DB_HOST:?DB_HOST не задан. Укажите в .env или переменной окружения}"
    : "${DB_PASSWORD:?DB_PASSWORD не задан}"
fi

DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-cam}"
DB_USER="${DB_USER:-cam}"
MQTT_HOST="${MQTT_HOST:-localhost}"
MQTT_PORT="${MQTT_PORT:-1883}"

log "=== Deploying cam-analyzer on Podman ==="
log "Pod:       $POD_NAME"
log "Data root: $DATA_ROOT"
log "DB:        $DB_HOST:$DB_PORT/$DB_NAME (local=$WITH_LOCAL_PG)"
log "MQTT:      $MQTT_HOST:$MQTT_PORT (local=$WITH_MQTT)"
log "GPU:       $WITH_GPU"
log "Systemd:   $WITH_SYSTEMD"

# ------------------------------------------------------------------
# 1. Проверка наличия podman и создание каталогов
# ------------------------------------------------------------------
command -v podman &>/dev/null || fail "Podman не установлен. Запустите scripts/install-deps.sh"

mkdir -p "${DATA_ROOT}/frames" "${DATA_ROOT}/models" \
         "${DATA_ROOT}/mqtt-config" "${DATA_ROOT}/mqtt-data" \
          "${DATA_ROOT}/pgdata" "${DATA_ROOT}/config" "${DATA_ROOT}/logs"

# Copy config files from project (always update)
for f in settings.yaml cameras.yaml triggers.yaml; do
    if [[ -f "${PROJECT_DIR}/config/$f" ]]; then
        cp "${PROJECT_DIR}/config/$f" "${DATA_ROOT}/config/"
        log "Copied config/$f → ${DATA_ROOT}/config/"
    fi
done

chown -R 1001:0 "${DATA_ROOT}/config" "${DATA_ROOT}/logs" "${DATA_ROOT}/frames" "${DATA_ROOT}/models"
chmod -R 775 "${DATA_ROOT}/config" "${DATA_ROOT}/logs" "${DATA_ROOT}/frames" "${DATA_ROOT}/models"
log "Data directories: ${DATA_ROOT}/"

# ------------------------------------------------------------------
# 2. Проверка/создание сети (host network для доступа к камерам)
# ------------------------------------------------------------------
NETWORK_NAME="cam-net"
if ! podman network exists "$NETWORK_NAME" 2>/dev/null; then
    podman network create "$NETWORK_NAME"
    log "Network '$NETWORK_NAME' created"
else
    log "Network '$NETWORK_NAME' exists"
fi

# ------------------------------------------------------------------
# 3. Остановка старого пода если есть
# ------------------------------------------------------------------
if podman pod exists "$POD_NAME" 2>/dev/null; then
    log "Stopping existing pod '$POD_NAME'..."
    podman pod stop "$POD_NAME" 2>/dev/null || true
    podman pod rm -f "$POD_NAME" 2>/dev/null || true
fi

# ------------------------------------------------------------------
# 4. Создание пода
# ------------------------------------------------------------------
POD_CREATE_ARGS=(
    --name "$POD_NAME"
    --hostname cam-analyzer
    --network "$NETWORK_NAME"
)
# Проброс портов наружу (health + опционально mqtt)
POD_CREATE_ARGS+=(--publish "${HEALTH_PORT:-8090}:${HEALTH_PORT:-8090}")
if [[ "$WITH_MQTT" == "yes" ]]; then
    POD_CREATE_ARGS+=(--publish 1883:1883)
fi

podman pod create "${POD_CREATE_ARGS[@]}"
log "Pod '$POD_NAME' created"

# ------------------------------------------------------------------
# 4.5 Проверка RTSP-камер (опционально, WITH_TEST_STREAM=yes)
# ------------------------------------------------------------------
if [[ "${WITH_TEST_STREAM:-no}" == "yes" ]]; then
    log "Testing RTSP streams from config..."
    if [[ -f "${PROJECT_DIR}/config/cameras.yaml" ]]; then
        bash "${SCRIPT_DIR}/test-stream.sh" --from-config || \
            warn "Some cameras may be unreachable — проверьте логи выше"
    else
        warn "cameras.yaml not found, skipping stream test"
    fi
fi

# ------------------------------------------------------------------
# 5. Запуск Mosquitto MQTT (опционально)
# ------------------------------------------------------------------
if [[ "$WITH_MQTT" == "yes" ]]; then
    log "Starting Mosquitto MQTT broker..."
    mkdir -p "${DATA_ROOT}/mqtt-data" "${DATA_ROOT}/mqtt-config"

    cat > "${DATA_ROOT}/mqtt-config/mosquitto.conf" <<'MQTT_EOF'
listener 1883 0.0.0.0
allow_anonymous true
persistence true
persistence_location /mosquitto/data
log_dest stdout
MQTT_EOF

    podman run -d --pod "$POD_NAME" --name mqtt-broker \
        -v "${DATA_ROOT}/mqtt-config/mosquitto.conf:/mosquitto/config/mosquitto.conf:Z" \
        -v "${DATA_ROOT}/mqtt-data:/mosquitto/data:Z" \
        --health-cmd "pgrep mosquitto > /dev/null || exit 1" \
        --health-interval 10s --health-timeout 5s --health-retries 3 \
        --health-start-period 10s \
        "$MQTT_IMAGE"
    log "Mosquitto started"
    MQTT_HOST="localhost"
    MQTT_PORT="1883"
fi

# ------------------------------------------------------------------
# 5.5. Запуск PostgreSQL (опционально, в том же поде)
# ------------------------------------------------------------------
if [[ "$WITH_LOCAL_PG" == "yes" ]]; then
    log "Starting PostgreSQL in pod..."

    PG_IMAGE="${PG_IMAGE:-docker.io/pgvector/pgvector:pg17}"
    PG_CONTAINER="${PG_CONTAINER:-cam-postgres}"

    mkdir -p "${DATA_ROOT}/pgdata"

    INIT_DIR="${PROJECT_DIR}/data/pg-init"
    mkdir -p "$INIT_DIR"

    cat > "$INIT_DIR/01-init.sql" <<'SQLA'
CREATE EXTENSION IF NOT EXISTS vector;
SQLA

    cat > "$INIT_DIR/02-migrations.sql" <<SQLB
CREATE TABLE IF NOT EXISTS cameras (
    id VARCHAR(64) PRIMARY KEY, name VARCHAR(256) NOT NULL,
    rtsp_url TEXT NOT NULL, fps INTEGER DEFAULT 10,
    enabled BOOLEAN DEFAULT true, config_json JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS tracked_objects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id VARCHAR(64) REFERENCES cameras(id),
    track_id INTEGER NOT NULL, class_name VARCHAR(32) NOT NULL,
    first_seen TIMESTAMPTZ DEFAULT now(), last_seen TIMESTAMPTZ DEFAULT now(),
    plate_number VARCHAR(32), face_id VARCHAR(64), face_hash VARCHAR(32),
    embedding VECTOR(512), metadata JSONB, name VARCHAR(128), appearance_count INTEGER DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_objects_camera_track ON tracked_objects(camera_id, track_id);
CREATE INDEX IF NOT EXISTS idx_objects_plate ON tracked_objects(plate_number) WHERE plate_number IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_objects_embedding ON tracked_objects
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100) WHERE embedding IS NOT NULL;

    CREATE TABLE IF NOT EXISTS crop_samples (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        camera_id VARCHAR(64) NOT NULL,
        class_name VARCHAR(32) NOT NULL,
        bbox_x1 INTEGER NOT NULL, bbox_y1 INTEGER NOT NULL,
        bbox_x2 INTEGER NOT NULL, bbox_y2 INTEGER NOT NULL,
        image_path VARCHAR(512) NOT NULL,
        phase VARCHAR(16) DEFAULT 'entry',
        is_val BOOLEAN DEFAULT false,
        timestamp TIMESTAMPTZ DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS idx_crop_samples_lookup ON crop_samples(camera_id, class_name, timestamp);

    CREATE TABLE IF NOT EXISTS frame_captures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    object_id UUID REFERENCES tracked_objects(id) ON DELETE CASCADE,
    image_path TEXT NOT NULL,
    bbox_x1 INTEGER NOT NULL, bbox_y1 INTEGER NOT NULL,
    bbox_x2 INTEGER NOT NULL, bbox_y2 INTEGER NOT NULL,
    confidence REAL DEFAULT 0.0, timestamp TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    object_id UUID REFERENCES tracked_objects(id),
    event_type VARCHAR(64) NOT NULL, trigger_name VARCHAR(128),
    action_result JSONB, timestamp TIMESTAMPTZ DEFAULT now()
);
SQLB

    podman run -d --pod "$POD_NAME" --name "$PG_CONTAINER" --restart unless-stopped \
        -e POSTGRES_DB="$DB_NAME" \
        -e POSTGRES_USER="$DB_USER" \
        -e POSTGRES_PASSWORD="$DB_PASSWORD" \
        -e POSTGRES_HOST_AUTH_METHOD=md5 \
        -v "${DATA_ROOT}/pgdata:/var/lib/postgresql/data:Z" \
        -v "${INIT_DIR}/01-init.sql:/docker-entrypoint-initdb.d/01-init.sql:Z" \
        -v "${INIT_DIR}/02-migrations.sql:/docker-entrypoint-initdb.d/02-migrations.sql:Z" \
        --health-cmd "pg_isready -U ${DB_USER} -d ${DB_NAME}" \
        --health-interval 5s --health-timeout 3s --health-retries 10 \
        --health-start-period 15s \
        "$PG_IMAGE"

    log "PostgreSQL container started, waiting for readiness..."
    for i in $(seq 1 30); do
        if podman exec "$PG_CONTAINER" pg_isready -U "$DB_USER" -d "$DB_NAME" &>/dev/null; then
            log "PostgreSQL ready"
            break
        fi
        if [[ $i -eq 30 ]]; then
            podman logs "$PG_CONTAINER" 2>&1 | tail -20
            fail "PostgreSQL failed to start within 60s"
        fi
        sleep 2
    done
    DB_HOST="localhost"
fi

# ------------------------------------------------------------------
# 6. Запуск cam-analyzer (build или pull)
# ------------------------------------------------------------------
if [[ "$WITH_BUILD" == "yes" ]]; then
    log "Building cam-analyzer image locally..."
    podman build -t "$APP_IMAGE" -f "${PROJECT_DIR}/docker/Containerfile.debian" "$PROJECT_DIR"
elif [[ -n "$REGISTRY" ]]; then
    APP_IMAGE="${REGISTRY}/cam-analyzer:${IMAGE_TAG}"
    log "Pulling cam-analyzer from registry: $APP_IMAGE"
    podman pull "$APP_IMAGE"
fi

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
    -e MQTT_HOST="$MQTT_HOST"
    -e MQTT_PORT="$MQTT_PORT"
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

# GPU проброс
if [[ "$WITH_GPU" == "yes" ]]; then
    APP_RUN_ARGS+=(--device nvidia.com/gpu=all --security-opt=label=disable)
fi

podman run "${APP_RUN_ARGS[@]}" \
        --health-cmd "python3 -c \"import urllib.request; r=urllib.request.urlopen('http://localhost:8090/health', timeout=3); import json; d=json.load(r); exit(0 if d.get('status') in ('healthy','degraded') else 1)\"" \
        --health-interval 15s --health-timeout 5s --health-retries 3 \
        --health-start-period 30s \
        "$APP_IMAGE"
log "cam-analyzer container started"

# Set SELinux labels for SELinux-enabled systems (RHEL/CentOS)
# Container runs as user 1001, so host directories need proper context
podman volume create cam-config 2>/dev/null || true

# ------------------------------------------------------------------
# 7. Генерация systemd unit (опционально)
# ------------------------------------------------------------------
if [[ "$WITH_SYSTEMD" == "yes" ]]; then
    log "Generating systemd unit files..."
    mkdir -p "${HOME}/.config/systemd/user"

    podman generate systemd --new --name --files "$POD_NAME"
    for f in pod-*.service; do
        mv "$f" "${HOME}/.config/systemd/user/"
    done

    if systemctl --user daemon-reload 2>/dev/null; then
        systemctl --user enable podman-pod-${POD_NAME}.service 2>/dev/null || true
        log "systemd units installed in ~/.config/systemd/user/"
        log "Для автозапуска: loginctl enable-linger $USER"
    else
        warn "systemd-user not available (no D-Bus session). Units saved to ~/.config/systemd/user/"
        warn "Используйте: loginctl enable-linger $USER && systemctl --user start podman-pod-${POD_NAME}.service"
    fi
fi

# ------------------------------------------------------------------
# 8. Health check
# ------------------------------------------------------------------
sleep 3
if podman pod ps --filter "name=$POD_NAME" --format "{{.Status}}" | grep -q "Running"; then
    log "=== Pod '$POD_NAME' running OK ==="
    podman pod ps --filter "name=$POD_NAME"
    echo ""
    log "Logs: podman logs cam-analyzer"
    log "Shell: podman exec -it cam-analyzer /bin/bash"
else
    log "=== Pod status check ==="
    podman pod ps --filter "name=$POD_NAME"
    podman logs cam-analyzer 2>&1 | tail -30
    fail "Pod failed to start"
fi
