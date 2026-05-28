# CAM Video Analytics

Система видеоаналитики с AI-обнаружением, трекингом и распознаванием на RHEL 9/10.

## Стек

| Компонент | Технология |
|-----------|-----------|
| Контейнеризация | Podman (rootless) |
| БД | PostgreSQL 14+ с pgvector |
| MQTT | Mosquitto |
| Обнаружение объектов | YOLOv11 (Ultralytics) |
| Трекинг | DeepSORT (Kalman + IoU) |
| Распознавание лиц | InsightFace (ArcFace) |
| Распознавание номеров | PaddleOCR |
| Базовый образ | UBI 9 `registry.access.redhat.com/ubi9/python-311` |

---

## Файлы проекта

```
cam/
├── config/
│   ├── cameras.yaml         — список RTSP-камер
│   ├── triggers.yaml        — триггеры и действия (webhook/mqtt/log)
│   └── settings.yaml        — пороги, модель YOLO, трекер
├── scripts/
│   ├── deploy.sh            — развёртывание в podman pod
│   ├── deploy-pg.sh         — PostgreSQL standalone
│   ├── build-push.sh        — сборка образа + push в registry
│   ├── test-stream.sh       — проверка RTSP/ONVIF камер
│   ├── install-deps.sh      — установка зависимостей на RHEL хосте
│   ├── install-systemd.sh   — systemd автозапуск
│   └── health-check.sh      — проверка работоспособности
├── docker/
│   ├── Containerfile         — основной (UBI 9)
│   ├── Containerfile.debian  — запасной (Debian slim)
│   └── podman-compose.yml    — compose-файл
├── src/                      — исходный код
└── .env.example              — шаблон переменных окружения
```

---

## Быстрый старт

### 1. Установка зависимостей хоста

```bash
bash scripts/install-deps.sh
```

Создаёт структуру `/srv/cam-analyzer/`, устанавливает podman, настраивает rootless, SELinux.

### 2. Конфигурация

```bash
cp .env.example .env && vim .env
```

### 2.5 Проверка камер (рекомендуется до деплоя)

```bash
# Проверить конкретную камеру по URL
bash scripts/test-stream.sh rtsp://admin:pass@192.168.10.18:554/stream

# Проверить все камеры из cameras.yaml
bash scripts/test-stream.sh --from-config

# Проверить конкретную камеру из конфига
bash scripts/test-stream.sh --from-config gate1
```

Скрипт проверяет: DNS, TCP-порт, RTSP DESCRIBE/200/401/403, читает видеопоток через ffprobe или OpenCV.

### 3. Запуск

```bash
WITH_LOCAL_PG=yes WITH_MQTT=yes bash scripts/deploy.sh
```

### 4. Проверка

```bash
bash scripts/health-check.sh
curl http://localhost:8090/health
```

---

## Переменные окружения (.env)

### Данные

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `DATA_ROOT` | `/srv/cam-analyzer` | Корень данных на хосте |

Структура внутри `DATA_ROOT`:
```
/srv/cam-analyzer/
├── frames/        — снимки объектов
├── models/        — кеш AI-моделей (YOLO, InsightFace)
├── pgdata/        — данные PostgreSQL
├── mqtt-config/   — конфиг Mosquitto (mosquitto.conf)
└── mqtt-data/     — persistence Mosquitto
```

### База данных

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `WITH_LOCAL_PG` | `no` | `yes` — поднять PG в том же pod |
| `DB_HOST` | — | Хост PostgreSQL (обязателен если `WITH_LOCAL_PG=no`) |
| `DB_PORT` | `5432` | Порт |
| `DB_NAME` | `cam` | Имя БД |
| `DB_USER` | `cam` | Пользователь |
| `DB_PASSWORD` | — | Пароль (авто-генерится при `WITH_LOCAL_PG=yes`) |

### MQTT

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `WITH_MQTT` | `no` | `yes` — запустить Mosquitto в поде |
| `MQTT_HOST` | `localhost` | Хост MQTT брокера |
| `MQTT_PORT` | `1883` | Порт |

### Образ

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `REGISTRY` | — | Адрес registry: `registry.example.com/project` |
| `IMAGE_TAG` | `latest` | Тег образа |
| `WITH_BUILD` | `yes` | `yes` — собрать локально, `no` — pull из registry |

### Pod

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `POD_NAME` | `cam-pod` | Имя podman pod |
| `HEALTH_PORT` | `8090` | Порт HTTP health endpoint |
| `WITH_GPU` | `no` | `yes` — пробросить NVIDIA GPU |
| `WITH_SYSTEMD` | `no` | `yes` — сгенерировать systemd unit при деплое |
| `TZ` | `Europe/Moscow` | Часовой пояс |

---

## Скрипты

### `scripts/deploy.sh` — основной деплой

Создаёт pod, запускает cam-analyzer, опционально PG и MQTT.

```bash
# Всё локально (PG + MQTT + cam в одном поде)
WITH_LOCAL_PG=yes WITH_MQTT=yes bash scripts/deploy.sh

# PG локально, MQTT снаружи
WITH_LOCAL_PG=yes bash scripts/deploy.sh

# Всё снаружи — только cam-analyzer
bash scripts/deploy.sh

# Из registry (без локальной сборки)
WITH_BUILD=no bash scripts/deploy.sh

# С GPU
WITH_GPU=yes bash scripts/deploy.sh

# Сгенерировать systemd unit
WITH_SYSTEMD=yes bash scripts/deploy.sh
```

### `scripts/deploy-pg.sh` — PostgreSQL standalone

Запускает PG+pgvector в отдельном контейнере (вне пода). Полезно когда PG должен быть доступен снаружи.

```bash
bash scripts/deploy-pg.sh                     # standalone контейнер
WITH_POD=cam-pod bash scripts/deploy-pg.sh    # внутри существующего пода
```

Выводит сгенерированный пароль и настройки для `.env`.

### `scripts/test-stream.sh` — проверка RTSP/ONVIF камер

```bash
bash scripts/test-stream.sh rtsp://admin:pass@192.168.10.18:554/stream
bash scripts/test-stream.sh --from-config               # все камеры из cameras.yaml
bash scripts/test-stream.sh --from-config gate1         # конкретная камера по id
bash scripts/test-stream.sh 192.168.10.18 554           # сканирование ONVIF
```

Проверяет: DNS → TCP-порт → RTSP DESCRIBE (200/401/403/404) → чтение кадра через ffprobe или OpenCV → ONVIF устройство.

### `scripts/build-push.sh` — сборка + push в registry

```bash
bash scripts/build-push.sh                    # UBI образ → push в REGISTRY
bash scripts/build-push.sh --debian           # Debian образ
bash scripts/build-push.sh --tag v1.2.0      # тегировать версией
bash scripts/build-push.sh --no-push          # только собрать, без push
bash scripts/build-push.sh --debian --no-push # собрать Debian локально
```

Требует `REGISTRY=...` в `.env`.

### `scripts/health-check.sh` — проверка

```bash
bash scripts/health-check.sh                  # разовая проверка
bash scripts/health-check.sh --watch          # мониторинг каждые 10 сек
bash scripts/health-check.sh --watch 5        # мониторинг каждые 5 сек
```

Проверяет: pod, контейнеры, HTTP `/health`, достижимость БД, последние логи.

### `scripts/install-deps.sh` — зависимости RHEL хоста

```bash
bash scripts/install-deps.sh
```

Устанавливает podman, опционально NVIDIA container toolkit, настраивает subuid/subgid для rootless, SELinux политики, создаёт `/srv/cam-analyzer/`.

### `scripts/install-systemd.sh` — systemd сервис

```bash
bash scripts/install-systemd.sh
```

Генерирует systemd user unit из запущенного пода для автозапуска при загрузке. Требует `loginctl enable-linger $USER`.

---

## Сценарии развёртывания

### A. Всё на одном хосте

```bash
WITH_LOCAL_PG=yes WITH_MQTT=yes bash scripts/deploy.sh
```

Результат: pod `cam-pod` с тремя контейнерами — postgres, mosquitto, cam-analyzer.
Все общаются через localhost. Данные в `/srv/cam-analyzer/`.

### B. PG на внешнем сервере

```bash
# На сервере с PG:
createdb cam && createuser cam
psql -c "CREATE EXTENSION vector" cam

# .env:
DB_HOST=10.0.0.50
DB_PASSWORD=strong_pass

bash scripts/deploy.sh
```

### C. Production из registry

```bash
# Разработчик: собрать и запушить
bash scripts/build-push.sh --tag v1.0.0

# Сервер: запустить из registry
REGISTRY=registry.example.com/pltech
IMAGE_TAG=v1.0.0
WITH_BUILD=no bash scripts/deploy.sh
```

---

## Конфигурация камер и триггеров

### `config/cameras.yaml`

```yaml
cameras:
  - id: "gate1"
    name: "Въездные ворота"
    rtsp_url: "rtsp://192.168.10.18:544/mode=real&idc=[1]&ids=[1]"
    fps: 10
    enabled: true
    roi: []                    # полигон зоны интереса, пусто = весь кадр
    motion_threshold: 0.15     # переопределяет глобальный порог
    motion_skip_seconds: 1.0   # мин. интервал между кадрами
```

### `config/triggers.yaml`

```yaml
triggers:
  - name: "alert_boss_car"
    type: plate                # plate | face | class
    values: ["A123BC", "M456OO"]
    actions:
      - type: webhook
        url: "http://alert-host:9000/alert"
      - type: mqtt
        topic: "cam/alerts"
      - type: log
        level: WARNING

  - name: "known_face"
    type: face
    source_db: true            # искать совпадения в БД по embedding
    actions:
      - type: log
        level: INFO
```

### `config/settings.yaml`

Глобальные настройки: модель YOLO (`yolo11n.pt` → `yolo11m.pt` для точности), пороги детекции, параметры трекера, motion detection, LPR, FaceRecog. Все значения переопределяются через env vars.

---

## Обновление

```bash
# Остановить
podman pod stop cam-pod && podman pod rm -f cam-pod

# Обновить код и перезапустить
git pull
bash scripts/deploy.sh

# Или из registry:
WITH_BUILD=no IMAGE_TAG=v1.1.0 bash scripts/deploy.sh
```

Данные в `/srv/cam-analyzer/pgdata`, `/srv/cam-analyzer/frames` сохраняются между перезапусками — podman volumes удаляются только при явном `podman volume rm`.

---

## Observability

### HTTP эндпоинты (порт 8090)

```bash
curl http://localhost:8090/health           # статус + сводка
curl http://localhost:8090/stats            # агрегированная статистика
curl http://localhost:8090/stats/detailed   # детальные счётчики
```

### Логи

```bash
podman logs cam-analyzer                    # все логи
podman logs -f cam-analyzer                 # tail
```

Каждые 30 секунд в лог пишется сводка:
```
STATS | cams=2/2 frames=1450 objects=87 events=3 errors=0
```

### Ручное управление

```bash
podman pod ps                               # статус пода
podman pod stop cam-pod                     # остановить
podman pod start cam-pod                    # запустить
podman exec -it cam-analyzer /bin/bash      # shell в контейнере
podman logs --tail 50 cam-postgres          # логи PG
```
