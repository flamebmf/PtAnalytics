# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
#!/bin/bash
# ============================================================
# deploy-pg.sh — поднятие PostgreSQL + pgvector в Podman
# Можно запустить отдельно или через deploy.sh (WITH_LOCAL_PG=yes)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_DIR}/.env"

# --- Параметры ---
PG_IMAGE="${PG_IMAGE:-docker.io/pgvector/pgvector:pg17}"
PG_CONTAINER="${PG_CONTAINER:-cam-postgres}"
DATA_ROOT="${DATA_ROOT:-/srv/cam-analyzer}"
WITH_POD="${WITH_POD:-}"                  # имя пода, куда добавить PG (опционально)
PG_HOST="${PG_HOST:-0.0.0.0}"
PG_LISTEN="${PG_LISTEN:-localhost}"       # что слушать наружу (0.0.0.0 для внешнего доступа)
PG_PUBLISH="${PG_PUBLISH:-5432}"          # порт наружу (5432 — стандартный)

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
fail() { echo "ERROR: $*" >&2; exit 1; }

# --- 0. Загрузка .env ---
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-cam}"
DB_USER="${DB_USER:-cam}"
DB_PASSWORD="${DB_PASSWORD:-cam_pass_$(openssl rand -hex 6)}"

log "=== Deploying PostgreSQL + pgvector ==="
log "Image:     $PG_IMAGE"
log "Container: $PG_CONTAINER"
log "Pod:       ${WITH_POD:-<standalone>}"
log "Data:      ${DATA_ROOT}/pgdata"
log "Database:  $DB_NAME / user=$DB_USER"

# --- 1. Проверка podman ---
command -v podman &>/dev/null || fail "Podman не установлен"

# --- 2. Удалить старый контейнер если есть ---
if podman container exists "$PG_CONTAINER" 2>/dev/null; then
    log "Removing existing container '$PG_CONTAINER'..."
    podman stop "$PG_CONTAINER" 2>/dev/null || true
    podman rm -f "$PG_CONTAINER" 2>/dev/null || true
fi

# --- 3. Создать каталог данных ---
mkdir -p "${DATA_ROOT}/pgdata"

# --- 4. Генерация init SQL для создания БД ---
INIT_DIR="${PROJECT_DIR}/data/pg-init"
mkdir -p "$INIT_DIR"

cat > "$INIT_DIR/01-init.sql" <<SQL_EOF
CREATE EXTENSION IF NOT EXISTS vector;
SQL_EOF

cat > "$INIT_DIR/02-migrations.sql" <<SQL_EOF
CREATE TABLE IF NOT EXISTS cameras (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(256) NOT NULL,
    rtsp_url TEXT NOT NULL,
    fps INTEGER DEFAULT 10,
    enabled BOOLEAN DEFAULT true,
    config_json JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tracked_objects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id VARCHAR(64) REFERENCES cameras(id),
    track_id INTEGER NOT NULL,
    class_name VARCHAR(32) NOT NULL,
    first_seen TIMESTAMPTZ DEFAULT now(),
    last_seen TIMESTAMPTZ DEFAULT now(),
    plate_number VARCHAR(32),
    face_id VARCHAR(64),
    face_hash VARCHAR(32),
    embedding VECTOR(512),
    metadata JSONB,
    name VARCHAR(128),
    appearance_count INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_objects_camera_track ON tracked_objects(camera_id, track_id);
CREATE INDEX IF NOT EXISTS idx_objects_plate ON tracked_objects(plate_number) WHERE plate_number IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_objects_face_id ON tracked_objects(face_id) WHERE face_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_objects_embedding ON tracked_objects USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100) WHERE embedding IS NOT NULL;

CREATE TABLE IF NOT EXISTS frame_captures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    object_id UUID REFERENCES tracked_objects(id) ON DELETE CASCADE,
    image_path TEXT NOT NULL,
    bbox_x1 INTEGER NOT NULL,
    bbox_y1 INTEGER NOT NULL,
    bbox_x2 INTEGER NOT NULL,
    bbox_y2 INTEGER NOT NULL,
    confidence REAL DEFAULT 0.0,
    timestamp TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_frames_object ON frame_captures(object_id);
CREATE INDEX IF NOT EXISTS idx_frames_ts ON frame_captures(timestamp);

CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    object_id UUID REFERENCES tracked_objects(id),
    event_type VARCHAR(64) NOT NULL,
    trigger_name VARCHAR(128),
    action_result JSONB,
    timestamp TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_events_object ON events(object_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp);
SQL_EOF

# --- 5. Сборка параметров запуска ---
RUN_ARGS=(
    -d
    --name "$PG_CONTAINER"
    --restart unless-stopped
    -e POSTGRES_DB="$DB_NAME"
    -e POSTGRES_USER="$DB_USER"
    -e POSTGRES_PASSWORD="$DB_PASSWORD"
    -e POSTGRES_HOST_AUTH_METHOD=md5
    -e POSTGRES_INITDB_ARGS="--auth-host=md5 --auth-local=trust"
    -v "${DATA_ROOT}/pgdata:/var/lib/postgresql/data:Z"
    -v "${INIT_DIR}/01-init.sql:/docker-entrypoint-initdb.d/01-init.sql:Z"
    -v "${INIT_DIR}/02-migrations.sql:/docker-entrypoint-initdb.d/02-migrations.sql:Z"
    --health-cmd "pg_isready -U ${DB_USER} -d ${DB_NAME}"
    --health-interval 5s
    --health-timeout 3s
    --health-retries 5
    --health-start-period 10s
)

# Если задан под — контейнер внутри пода
if [[ -n "$WITH_POD" ]]; then
    if ! podman pod exists "$WITH_POD" 2>/dev/null; then
        fail "Pod '$WITH_POD' не существует. Создайте его сначала."
    fi
    RUN_ARGS+=(--pod "$WITH_POD")
else
    # Иначе standalone — публикуем порт наружу
    RUN_ARGS+=(--publish "${PG_LISTEN}:${PG_PUBLISH}:5432")
fi

podman run "${RUN_ARGS[@]}" "$PG_IMAGE"
log "PostgreSQL container started"

# --- 6. Ждём готовности ---
log "Waiting for PostgreSQL to be ready..."
ATTEMPTS=0
MAX_ATTEMPTS=30
until podman exec "$PG_CONTAINER" pg_isready -U "$DB_USER" -d "$DB_NAME" &>/dev/null; do
    sleep 2
    ATTEMPTS=$((ATTEMPTS + 1))
    if [[ $ATTEMPTS -ge $MAX_ATTEMPTS ]]; then
        log "PG status:"
        podman logs "$PG_CONTAINER" 2>&1 | tail -20
        fail "PostgreSQL failed to start within ${MAX_ATTEMPTS}s"
    fi
    log "  ... waiting (${ATTEMPTS}/${MAX_ATTEMPTS})"
done

log "PostgreSQL is ready"

# --- 7. Вывод информации ---
echo ""
echo "=== PostgreSQL deployed ==="
echo "Container:  $PG_CONTAINER"
echo "Database:   $DB_NAME"
echo "User:       $DB_USER"
echo "Password:   $DB_PASSWORD"

if [[ -n "$WITH_POD" ]]; then
    echo "Access:     localhost:5432 (inside pod '$WITH_POD')"
else
    echo "Access:     localhost:${PG_PUBLISH}"
fi

echo ""
echo "# Сохраните в .env или передайте переменными:"
echo "DB_HOST=${WITH_POD:+localhost}"
echo "DB_PORT=5432"
echo "DB_NAME=${DB_NAME}"
echo "DB_USER=${DB_USER}"
echo "DB_PASSWORD=${DB_PASSWORD}"
