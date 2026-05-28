#!/bin/bash
# ============================================================
# install-systemd.sh — установка cam-analyzer как systemd сервиса
# Требует предварительно запущенный deploy.sh
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POD_NAME="${POD_NAME:-cam-pod}"

if ! podman pod exists "$POD_NAME" 2>/dev/null; then
    echo "ERROR: Pod '$POD_NAME' не найден. Запустите сначала scripts/deploy.sh"
    exit 1
fi

echo "=== Установка systemd сервиса для podman pod '$POD_NAME' ==="

# Генерация unit-файлов
mkdir -p "${HOME}/.config/systemd/user"

cd /tmp
podman generate systemd --new --files --name "$POD_NAME"

# Перемещаем с правильными именами
for f in pod-*.pod; do
    target="${HOME}/.config/systemd/user/podman-pod-${POD_NAME}.pod"
    mv "$f" "$target"
    echo "Created: $target"
done

for f in container-*.container; do
    # container-cam-analyzer.service и container-mqtt-broker.service
    short=$(echo "$f" | sed 's/container-//; s/-container//')
    target="${HOME}/.config/systemd/user/${short}"
    mv "$f" "$target"
    echo "Created: $target"
done

# Перезагрузка systemd
systemctl --user daemon-reload

# Включаем pod (автозапуск контейнеров)
systemctl --user enable podman-pod-${POD_NAME}.service 2>/dev/null || true

# Включаем linger чтобы сервисы работали без логина
echo ""
echo ">>> Для автозапуска при загрузке системы выполните:"
echo "    loginctl enable-linger \$USER"
echo ""
echo ">>> Управление сервисом:"
echo "    systemctl --user start cam-analyzer.service"
echo "    systemctl --user stop  cam-analyzer.service"
echo "    systemctl --user status cam-analyzer.service"
echo "    journalctl --user -u cam-analyzer.service -f"
