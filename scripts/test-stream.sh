# Copyright (c) 2026 PluromTech.com
# SPDX-License-Identifier: GPL-3.0-only
#!/bin/bash
# ============================================================
# test-stream.sh — проверка RTSP/ONVIF через podman контейнеры
# Требуется только podman. Всё остальное — внутри контейнеров.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_DIR}/.env"

TESTER_IMAGE="${TESTER_IMAGE:-docker.io/jrottenberg/ffmpeg:7.1-ubuntu}"
APP_IMAGE="${APP_IMAGE:-cam-analyzer:latest}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}[OK]${NC}     $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC}   $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC}   $1"; }
info() { echo -e "  ${CYAN}[INFO]${NC}   $1"; }

parse_rtsp_url() {
    local url="$1"
    url="${url#rtsp://}"
    HOST="${url%%[:/@]*}"
    local rest="${url#$HOST}"
    PORT="${rest#:}"; PORT="${PORT%%[/?@]*}"
    [[ "$PORT" =~ ^[0-9]+$ ]] || PORT=554
    PATH_PART="/${rest#*/}"; PATH_PART="${PATH_PART%%\'*}"
    AUTH=""
    if [[ "$rest" == *"@"* ]]; then
        AUTH="${rest%%@*}"
        [[ "$AUTH" == *":"* ]] || AUTH=""
    fi
}

# ------------------------------------------------------------------
# Тест внутри контейнера ffmpeg: TCP → RTSP handshake → ffprobe
# ------------------------------------------------------------------
test_in_container() {
    local url="$1" host="$2" port="$3" path="$4" auth="$5"

    info "Testing $url via podman container..."

    podman run --rm --network host \
        -e TEST_URL="$url" -e TEST_HOST="$host" -e TEST_PORT="$port" \
        -e TEST_PATH="$path" -e TEST_AUTH="$auth" \
        "$TESTER_IMAGE" sh -c '
# TCP check
if timeout 5 nc -zv "$TEST_HOST" "$TEST_PORT" 2>/dev/null; then
    echo "[OK]     TCP connected: $TEST_HOST:$TEST_PORT"
else
    echo "[FAIL]   TCP unreachable: $TEST_HOST:$TEST_PORT"
    exit 1
fi

# RTSP DESCRIBE handshake
AUTH_HDR=""
if [ -n "$TEST_AUTH" ]; then
    AUTH_HDR="Authorization: Basic $(echo -n "$TEST_AUTH" | base64 -w0)\r\n"
fi

REQUEST="DESCRIBE rtsp://${TEST_HOST}:${TEST_PORT}${TEST_PATH} RTSP/1.0\r\nCSeq: 1\r\nUser-Agent: cam-test\r\n${AUTH_HDR}\r\n"
RESP=$(echo -ne "$REQUEST" | timeout 5 nc "$TEST_HOST" "$TEST_PORT" 2>/dev/null || true)

if echo "$RESP" | grep -q "RTSP/1.0 200"; then
    echo "[OK]     RTSP DESCRIBE: 200 OK"
    echo "$RESP" | grep -q "m=video" && echo "[INFO]   SDP video track detected"
elif echo "$RESP" | grep -qE "RTSP/1.0 401|RTSP/1.0 403"; then
    if [ -z "$TEST_AUTH" ]; then
        echo "[WARN]   RTSP: 401/403 — требуется авторизация"
        echo "[INFO]   Добавьте в URL: rtsp://user:password@${TEST_HOST}:${TEST_PORT}${TEST_PATH}"
    else
        echo "[WARN]   RTSP: 401/403 — неверные учётные данные"
    fi
    exit 0
elif echo "$RESP" | grep -q "RTSP/1.0 404"; then
    echo "[FAIL]   RTSP: 404 — путь не найден: ${TEST_PATH}"
    exit 1
elif echo "$RESP" | grep -q "RTSP/1.0"; then
    CODE=$(echo "$RESP" | grep -oi "RTSP/1.0 [0-9]*" | head -1)
    echo "[WARN]   RTSP: ${CODE:-unknown response}"
else
    echo "[WARN]   RTSP: нет RTSP-ответа (возможно не-RTSP порт)"
    echo "$RESP" | head -1 | sed "s/^/  /"
fi

# ffprobe — получить видеопотоки
echo "[INFO]   probing streams via ffprobe..."
OUT=$(timeout 15 ffprobe -rtsp_transport tcp -timeout 5000000 \
    -v quiet -print_format json -show_streams -i "$TEST_URL" 2>&1) || true

if echo "$OUT" | grep -q '"codec_type": "video"'; then
    INFO=$(echo "$OUT" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    for s in d.get(\"streams\",[]):
        if s.get(\"codec_type\")==\"video\":
            print(f\"{s.get(\"width\",\"?\")}x{s.get(\"height\",\"?\")} {s.get(\"codec_name\",\"?\")}\", end=\"\")
            break
except: pass
" 2>/dev/null || echo "video stream")
    echo "[OK]     ffprobe: $INFO"
else
    echo "[WARN]   ffprobe: видеопоток не обнаружен — проверьте авторизацию или формат"
    echo "$OUT" | tail -3 | sed "s/^/  /"
fi
'
    return ${PIPESTATUS[0]}
}

# ------------------------------------------------------------------
# OpenCV тест через cam-analyzer образ (если уже собран)
# ------------------------------------------------------------------
test_opencv_image() {
    local url="$1"
    if podman image exists "$APP_IMAGE" 2>/dev/null; then
        info "Testing via OpenCV (cam-analyzer image)…"
        podman run --rm --network host "$APP_IMAGE" python3 - "$url" <<'PYEOF'
import sys, cv2, time
url = sys.argv[1]
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("[FAIL]   OpenCV: failed to open stream")
    sys.exit(1)
for _ in range(5):
    ret, frame = cap.read()
    if ret and frame is not None:
        h, w = frame.shape[:2]
        t = time.time()
        print(f"[OK]     OpenCV: {w}x{h}, first frame read OK")
        cap.release()
        sys.exit(0)
    time.sleep(0.5)
print("[FAIL]   OpenCV: no frame received in ~3s")
cap.release()
sys.exit(1)
PYEOF
    else
        info "OpenCV test skipped (image $APP_IMAGE not built locally)"
    fi
}

# ------------------------------------------------------------------
# ONVIF сканирование
# ------------------------------------------------------------------
test_onvif() {
    local host="$1"
    info "Scanning ONVIF on $host…"
    podman run --rm --network host -e H="$host" "$TESTER_IMAGE" sh -c '
for p in 80 8000 8080 8899; do
    code=$(timeout 3 curl -s -o /dev/null -w "%{http_code}" "http://${H}:$p/onvif/device_service" 2>/dev/null)
    case "$code" in 200|401|405)
        echo "[OK]     ONVIF: http://${H}:$p/onvif/device_service (HTTP $code)"
        exit 0 ;;
    esac
done
echo "[WARN]   ONVIF not found (ports 80,8000,8080,8899)"
'
}

# ==================================================================
# Main
# ==================================================================

echo ""
echo "=== RTSP/ONVIF Stream Test  $(date '+%H:%M:%S') ==="
echo ""

# Pull тестового образа
if ! podman image exists "$TESTER_IMAGE" 2>/dev/null; then
    info "Pulling $TESTER_IMAGE ..."
    podman pull "$TESTER_IMAGE" || warn "Failed to pull test image"
fi

# --- Из конфига ---
if [[ "${1:-}" == "--from-config" ]]; then
    CAM_FILE="${PROJECT_DIR}/config/cameras.yaml"
    [[ -f "$CAM_FILE" ]] || { fail "cameras.yaml not found: $CAM_FILE"; exit 1; }

    FILTER_ID="${2:-}"

    # Парсим YAML внутри контейнера (stdlib, без PyYAML)
    CAMERAS=$(podman run --rm \
        -v "${PROJECT_DIR}/config:/config:ro" \
        "$TESTER_IMAGE" python3 -c "
import re, sys
filter_id = sys.argv[1] if len(sys.argv) > 1 else ''
with open('/config/cameras.yaml') as f:
    text = f.read()
for block in re.split(r'\n- ', text):
    m_id   = re.search(r'id:\s*\"([^\"]*)\"', block)
    m_name = re.search(r'name:\s*\"([^\"]*)\"', block)
    m_url  = re.search(r'rtsp_url:\s*\"([^\"]*)\"', block)
    if m_id and m_url:
        name = m_name.group(1) if m_name else ''
        if not filter_id or m_id.group(1) == filter_id:
            print(f"{m_id.group(1)}|{name}|{m_url.group(1)}")
" "$FILTER_ID" 2>/dev/null)


    [[ -n "$CAMERAS" ]] || { fail "No cameras in config"; exit 1; }

    echo "Found $(echo "$CAMERAS" | wc -l) camera(s) in config"
    echo ""

    while IFS='|' read -r cam_id cam_name cam_url; do
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Camera: $cam_id ($cam_name)"
        echo "URL:    $cam_url"
        echo ""
        parse_rtsp_url "$cam_url"
        test_in_container "$cam_url" "$HOST" "$PORT" "$PATH_PART" "$AUTH" || true
        test_opencv_image "$cam_url" || true
        echo ""
    done <<< "$CAMERAS"

# --- ONVIF ---
elif [[ "${1:-}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] && [[ "${2:-}" =~ ^[0-9]+$ ]] && [[ "${3:-}" != rtsp* ]]; then
    test_onvif "$1"
    echo ""

# --- Один URL ---
elif [[ "${1:-}" == rtsp://* ]] || [[ "${1:-}" == rtmp://* ]]; then
    URL="$1"
    echo "URL: $URL"
    echo ""
    parse_rtsp_url "$URL"
    [[ -n "$AUTH" ]] && info "Auth: $AUTH"
    test_in_container "$URL" "$HOST" "$PORT" "$PATH_PART" "$AUTH" || true
    test_opencv_image "$URL" || true
    echo ""

else
    echo "Usage:"
    echo "  bash scripts/test-stream.sh rtsp://admin:pass@192.168.10.18:554/stream"
    echo "  bash scripts/test-stream.sh --from-config"
    echo "  bash scripts/test-stream.sh --from-config gate1"
    echo "  bash scripts/test-stream.sh 192.168.10.18 554"
    echo ""
    echo "Все проверки в podman-контейнере ($TESTER_IMAGE)."
    echo "Хост-утилиты не требуются — нужен только podman."
    exit 1
fi

echo "=== Done ==="
