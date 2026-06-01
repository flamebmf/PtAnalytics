# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
#!/bin/bash
# ============================================================
# health-check.sh — проверка состояния cam-analyzer
# Использование: bash scripts/health-check.sh [--watch]
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_DIR}/.env"

if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

POD_NAME="${POD_NAME:-cam-pod}"
APP_CONTAINER="${APP_CONTAINER:-cam-analyzer}"
PG_CONTAINER="${PG_CONTAINER:-cam-postgres}"
MQTT_CONTAINER="${MQTT_CONTAINER:-mqtt-broker}"
HEALTH_PORT="${HEALTH_PORT:-8090}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }

check() {
    echo ""
    echo "=== cam-analyzer health check $(date '+%Y-%m-%d %H:%M:%S') ==="

    # Pod status
    if podman pod exists "$POD_NAME" 2>/dev/null; then
        POD_STATUS=$(podman pod ps --filter "name=$POD_NAME" --format "{{.Status}}" 2>/dev/null)
        if echo "$POD_STATUS" | grep -q "Running"; then
            ok "Pod '$POD_NAME': $POD_STATUS"
        else
            fail "Pod '$POD_NAME': $POD_STATUS"
        fi
    else
        fail "Pod '$POD_NAME' not found"
        return 1
    fi

    # Container status
    for cnt in "$APP_CONTAINER" "$PG_CONTAINER" "$MQTT_CONTAINER"; do
        if podman container exists "$cnt" 2>/dev/null; then
            CNT_STATUS=$(podman ps --filter "name=$cnt" --format "{{.Status}}" 2>/dev/null)
            if echo "$CNT_STATUS" | grep -qi "Up"; then
                ok "Container '$cnt': running"
            else
                warn "Container '$cnt': not running (state=$CNT_STATUS)"
            fi
        fi
    done

    # HTTP health endpoint
    if curl -sf "http://localhost:${HEALTH_PORT}/health" -o /tmp/cam-health.json 2>/dev/null; then
        STATUS=$(jq -r '.status' /tmp/cam-health.json 2>/dev/null)
        CAMS=$(jq -r '.cameras_running' /tmp/cam-health.json 2>/dev/null)
        FRAMES=$(jq -r '.total_frames_processed' /tmp/cam-health.json 2>/dev/null)
        OBJS=$(jq -r '.total_objects_stored' /tmp/cam-health.json 2>/dev/null)

        case "$STATUS" in
            healthy) ok "Health endpoint: $STATUS (cams=$CAMS, frames=$FRAMES, objects=$OBJS)" ;;
            degraded) warn "Health endpoint: $STATUS (cams=$CAMS, frames=$FRAMES, objects=$OBJS)" ;;
            *) fail "Health endpoint: $STATUS" ;;
        esac
    else
        fail "Health endpoint unreachable: http://localhost:${HEALTH_PORT}/health"
    fi

    # DB check (direct)
    if [[ -n "${DB_HOST:-}" && -n "${DB_PASSWORD:-}" ]]; then
        if podman exec "$POD_NAME" --latest sh -c "pg_isready -h ${DB_HOST} -p ${DB_PORT:-5432} -U ${DB_USER:-cam} -d ${DB_NAME:-cam}" 2>/dev/null; then
            ok "Database reachable: ${DB_HOST}:${DB_PORT:-5432}/${DB_NAME:-cam}"
        else
            warn "Database not reachable (may be external or local PG container)"
        fi
    fi

    # GPU (if applicable)
    if nvidia-smi &>/dev/null 2>&1; then
        ok "GPU detected"
    fi

    echo ""
    echo "Recent logs (last 5 lines):"
    podman logs --tail 5 "$APP_CONTAINER" 2>/dev/null | sed 's/^/  /' || echo "  (no logs)"
}

# Main
check

if [[ "${1:-}" == "--watch" ]]; then
    INTERVAL="${2:-10}"
    echo ""
    echo "Watching every ${INTERVAL}s, Ctrl+C to stop"
    while true; do
        sleep "$INTERVAL"
        check
    done
fi
