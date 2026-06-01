# Copyright (c) 2026 PlurumTech.com
# SPDX-License-Identifier: GPL-3.0-only
#!/bin/bash
# ============================================================
# install-deps.sh — установка зависимостей на RHEL 9/10 хосте
# ============================================================
set -euo pipefail

if [[ $EUID -eq 0 ]]; then
    echo "Запускайте без sudo, скрипт запросит права когда нужно"
    exit 1
fi

echo "=== Установка зависимостей для cam-analyzer (RHEL 9/10) ==="

# -- Podman --
if ! command -v podman &>/dev/null; then
    echo ">>> Установка podman..."
    sudo dnf install -y podman
fi

PODMAN_VER=$(podman --version)
echo "Podman: $PODMAN_VER"

# -- NVIDIA Container Toolkit (опционально) --
read -r -p "Установить NVIDIA Container Toolkit для GPU? [y/N]: " INSTALL_NVIDIA
if [[ "$INSTALL_NVIDIA" =~ ^[Yy]$ ]]; then
    if ! command -v nvidia-container-cli &>/dev/null; then
        echo ">>> Установка NVIDIA Container Toolkit..."
        OS_VERSION=$(rpm -E %rhel)
        curl -s -L "https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo" \
            | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
        sudo dnf install -y nvidia-container-toolkit

        # Генерация конфигурации CDI для podman
        sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
        echo "NVIDIA toolkit установлен. CDI конфигурация: /etc/cdi/nvidia.yaml"
    else
        echo "NVIDIA Container Toolkit уже установлен"
    fi
fi

# -- Добавление пользователя в subuid/subgid для rootless podman --
SUBUIDS=$(grep "^$(whoami):" /etc/subuid 2>/dev/null || true)
if [[ -z "$SUBUIDS" ]]; then
    echo ">>> Настройка rootless podman (subuid/subgid)..."
    UID_START=100000
    echo "$(whoami):${UID_START}:65536" | sudo tee -a /etc/subuid > /dev/null
    echo "$(whoami):${UID_START}:65536" | sudo tee -a /etc/subgid > /dev/null
    podman system migrate
    echo "subuid/subgid настроены"
fi

# -- SELinux: разрешаем монтирование томов в контейнерах --
if command -v getenforce &>/dev/null && [[ "$(getenforce)" != "Disabled" ]]; then
    echo ">>> SELinux: включаем container_manage_cgroup..."
    sudo setsebool -P container_manage_cgroup on 2>/dev/null || true
fi

# -- Директории данных --
DATA_ROOT="${DATA_ROOT:-/srv/cam-analyzer}"
mkdir -p "${DATA_ROOT}/frames" "${DATA_ROOT}/models" "${DATA_ROOT}/pgdata" \
         "${DATA_ROOT}/mqtt-data" "${DATA_ROOT}/mqtt-config"

if [[ $(id -u) -eq 0 ]]; then
    chown -R "$(logname):$(logname)" "${DATA_ROOT}" 2>/dev/null || true
fi
chcon -Rt container_file_t "${DATA_ROOT}" 2>/dev/null || true

echo ""
echo "=== Установка завершена ==="
echo "Версия podman:  $(podman --version)"
echo "GPU toolkit:    $(nvidia-container-cli --version 2>/dev/null || echo 'не установлен')"
echo ""
echo "Далее:"
echo "  1. Создайте .env файл на основе .env.example"
echo "  2. Выполните: bash scripts/deploy.sh"
