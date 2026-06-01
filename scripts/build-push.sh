# Copyright (c) 2026 PluromTech.com
# SPDX-License-Identifier: GPL-3.0-only
#!/bin/bash
# ============================================================
# build-push.sh — сборка образа и push в registry
# Использование:
#   bash scripts/build-push.sh              # собрать + push в registry из .env
#   bash scripts/build-push.sh --no-push    # только собрать, без push
#   bash scripts/build-push.sh --ubi9       # собрать UBI9-вариант
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_DIR}/.env"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
fail() { echo "ERROR: $*" >&2; exit 1; }

if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

REGISTRY="${REGISTRY:-}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DOCKERFILE="docker/Containerfile.debian"
DO_PUSH=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-push) DO_PUSH=false ;;
        --ubi9)    DOCKERFILE="docker/Containerfile"; IMAGE_TAG="${IMAGE_TAG}-ubi9" ;;
        --tag)     IMAGE_TAG="$2"; shift ;;
        *)         log "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$REGISTRY" ]]; then
    fail "REGISTRY не задан. Укажите в .env или экспортом: REGISTRY=registry.example.com/project"
fi

FULL_IMAGE="${REGISTRY}/cam-analyzer:${IMAGE_TAG}"

log "=== Build & Push cam-analyzer ==="
log "Dockerfile: $DOCKERFILE"
log "Registry:   $REGISTRY"
log "Tag:        $IMAGE_TAG"
log "Image:      $FULL_IMAGE"

# --- Build ---
log "Building..."
podman build -t "$FULL_IMAGE" -f "${PROJECT_DIR}/${DOCKERFILE}" "$PROJECT_DIR"

# --- Tag latest ---
if [[ "$IMAGE_TAG" != "latest" ]]; then
    LATEST="${REGISTRY}/cam-analyzer:latest"
    podman tag "$FULL_IMAGE" "$LATEST"
    log "Tagged: $LATEST"
fi

# --- Push ---
if $DO_PUSH; then
    log "Pushing $FULL_IMAGE..."
    podman push "$FULL_IMAGE"
    log "Push complete"

    if [[ "$IMAGE_TAG" != "latest" ]]; then
        podman push "$LATEST"
        log "Pushed latest tag"
    fi
else
    log "Skipping push (--no-push)"
fi

echo ""
echo "=== Done ==="
echo "Image: $FULL_IMAGE"
echo ""
echo "Для деплоя из registry:"
echo "  WITH_BUILD=no bash scripts/deploy.sh"
