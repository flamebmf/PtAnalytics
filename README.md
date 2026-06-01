# CAM Video Analytics

AI video analytics with object detection, tracking, face/LPR recognition, and user-guided fine-tuning.

---

## Stack

| Component | Technology |
|-----------|-----------|
| Containerization | Podman (RHEL) / Docker |
| Database | PostgreSQL 17 with pgvector |
| MQTT | Mosquitto |
| Object detection | YOLOv11 (Ultralytics) |
| Tracking | DeepSORT (Kalman + IoU) |
| Face recognition | InsightFace (ArcFace) |
| License plates | PaddleOCR |
| Vehicle make | CLIP zero-shot |
| UI | Bootstrap 5, vanilla JS |

## Architecture

Two-model approach:
- **YoloDetector** — base yolo11m (COCO 80 classes) — runs only on motion frames
- **CropClassifier** — fine-tuned.pt (custom classes) — classifies cropped objects

COCO detection and name classification run on crops. Training is done separately; resulting `fine-tuned.pt` is uploaded to the server.

## Quick Start

### Docker

```bash
# 1. Copy and configure
cp config/cameras.yaml.example config/cameras.yaml
cp config/settings.yaml.example config/settings.yaml
cp config/triggers.yaml.example config/triggers.yaml
cp .env.example .env
# edit .env

# 2. Run
docker compose up -d

# 3. Check
curl http://localhost:8090/health
```

### Podman (RHEL 9/10)

```bash
bash scripts/install-deps.sh
cp .env.example .env && vim .env
WITH_LOCAL_PG=yes WITH_MQTT=yes bash scripts/deploy.sh
curl http://localhost:8090/health
```

## Configuration

All configs in `config/`:
- `cameras.yaml` — RTSP cameras (copy from .example, set your rtsp_url)
- `settings.yaml` — detection thresholds, motion, tracker, LPR, face, VMR
- `triggers.yaml` — triggers and actions (webhook/mqtt/log)

Most settings are available via the web UI (gear icon in top right). Changes apply via `PUT /config` with hot-reload — no restart needed.

## UI Modes

### View 👁

- Image 70% + compact table 30%
- Only name and last seen time
- Suitable for monitoring

### Edit ✏

- Full table with detail panel on the right
- Search / filters / grouping
- Rename objects, assign classes
- Ignore and delete

Toggle button in the top right of the navbar.

## Fine-Tuning

1. Assign names to objects via UI
2. In "Дообучение" tab (Settings → 6th tab) download dataset ZIP
3. On a machine with GPU: run `training/train.bat`
4. Replace `fine-tuned.pt` on the server
5. Click "Загрузить модель" in UI or restart the container

`train.py` automatically creates a combined dataset from all named classes.

## Environment Variables (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | `postgres` (docker) | PostgreSQL host |
| `DB_PORT` | `5432` | Port |
| `DB_NAME` | `cam` | Database name |
| `DB_USER` | `cam` | User |
| `DB_PASSWORD` | `change_me` | Password |
| `MQTT_HOST` | `mosquitto` (docker) | MQTT host |
| `MQTT_PORT` | `1883` | MQTT port |
| `HEALTH_PORT` | `8090` | Health endpoint port |
| `TZ` | `Europe/Moscow` | Time zone |
| `WITH_LOCAL_PG` | `no` | Podman: PG in same pod |
| `WITH_GPU` | `no` | Podman: NVIDIA GPU passthrough |
| `OMP_NUM_THREADS` | `4` | CPU threads for OpenMP |

## API

### Endpoints (port 8090)

```bash
curl http://localhost:8090/health              # status + summary
curl http://localhost:8090/stats               # aggregated stats
curl http://localhost:8090/objects?limit=20    # object list
curl http://localhost:8090/objects/{id}        # object details
curl http://localhost:8090/filters             # available filters
curl http://localhost:8090/config              # config (GET/PUT)
curl http://localhost:8090/config/reload       # hot-reload
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/deploy.sh` | Deploy podman pod |
| `scripts/build-push.sh` | Build + push to registry |
| `scripts/test-stream.sh` | Check RTSP/ONVIF cameras |
| `scripts/health-check.sh` | Health monitoring |
| `training/train.py` | Fine-tune YOLO |
| `training/train.bat` | Same (Windows) |

## Project Structure

```
cam/
├── config/                # configuration
│   ├── cameras.yaml       # RTSP cameras
│   ├── settings.yaml      # global settings
│   ├── triggers.yaml      # triggers
│   └── mqtt/              # Mosquitto config
├── docker/                # Containerfiles
├── src/                   # source code
│   ├── detection/         # YOLO + CropClassifier
│   ├── recognition/       # LPR, Face, VMR
│   ├── tracking/          # DeepSORT
│   ├── storage/           # PostgreSQL + repository
│   ├── pipeline.py        # main pipeline
│   └── main.py            # aiohttp server
├── training/              # fine-tuning
├── ui/                    # web interface
├── docker-compose.yml     # Docker Compose
└── .env.example           # env template
```

---

# CAM Video Analytics

Система видеоаналитики с AI-обнаружением, трекингом, распознаванием лиц/номеров и дообучением на пользовательских данных.

## Стек

| Компонент | Технология |
|-----------|-----------|
| Контейнеризация | Podman (RHEL) / Docker |
| БД | PostgreSQL 17 с pgvector |
| MQTT | Mosquitto |
| Обнаружение объектов | YOLOv11 (Ultralytics) |
| Трекинг | DeepSORT (Kalman + IoU) |
| Распознавание лиц | InsightFace (ArcFace) |
| Распознавание номеров | PaddleOCR |
| Распознавание марок | CLIP zero-shot |
| UI | Bootstrap 5, vanilla JS |

## Архитектура

Проект использует **двухмодельный подход**:
- **YoloDetector** — базовая yolo11m (COCO 80 классов) — работает только на кадрах с движением
- **CropClassifier** — fine-tuned.pt (пользовательские классы) — классифицирует вырезанные объекты

Детекция COCO и классификация имён выполняются на crop'ах. Дообучение производится отдельно, результат (fine-tuned.pt) загружается на сервер.

## Быстрый старт

### Docker

```bash
# 1. Скопировать и настроить конфиги
cp config/cameras.yaml.example config/cameras.yaml
cp config/settings.yaml.example config/settings.yaml
cp config/triggers.yaml.example config/triggers.yaml
cp .env.example .env
# заполнить .env

# 2. Запустить
docker compose up -d

# 3. Проверить
curl http://localhost:8090/health
```

### Podman (RHEL 9/10)

```bash
bash scripts/install-deps.sh
cp .env.example .env && vim .env
WITH_LOCAL_PG=yes WITH_MQTT=yes bash scripts/deploy.sh
curl http://localhost:8090/health
```

## Конфигурация

Все конфиги в `config/`:
- `cameras.yaml` — RTSP-камеры (копия из .example, вписать свои rtsp_url)
- `settings.yaml` — пороги детекции, motion, трекер, LPR, face, VMR
- `triggers.yaml` — триггеры и действия (webhook/mqtt/log)

Большинство настроек доступно через веб-интерфейс (шестерёнка в правом верхнем углу). Изменения применяются через `PUT /config` и hot-reload — без перезапуска.

## Режимы UI

### Просмотр 👁

- Картинка 70% + компактная таблица 30%
- Только имя и время последнего появления
- Подходит для мониторинга

### Редактирование ✏

- Полная таблица с детальной панелью справа
- Поиск / фильтры / группировка
- Переименование объектов, назначение классов
- Игнорирование и удаление

Переключение — кнопка в правом верхнем углу навбара.

## Дообучение

1. Через UI задать имена объектам
2. На вкладке "Дообучение" (Settings → 6-я вкладка) скачать ZIP с датасетом
3. На машине с GPU: `training/train.bat`
4. Полученный `fine-tuned.pt` заменить на сервере
5. Через UI нажать "Загрузить модель" или перезапустить контейнер

`train.py` автоматически создаёт единый датасет из всех именованных классов.

## Переменные окружения (.env)

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `DB_HOST` | `postgres` (docker) | Хост PostgreSQL |
| `DB_PORT` | `5432` | Порт |
| `DB_NAME` | `cam` | Имя БД |
| `DB_USER` | `cam` | Пользователь |
| `DB_PASSWORD` | `change_me` | Пароль |
| `MQTT_HOST` | `mosquitto` (docker) | Хост MQTT |
| `MQTT_PORT` | `1883` | Порт MQTT |
| `HEALTH_PORT` | `8090` | Порт health endpoint |
| `TZ` | `Europe/Moscow` | Часовой пояс |
| `WITH_LOCAL_PG` | `no` | Podman: PG в том же pod |
| `WITH_GPU` | `no` | Podman: проброс NVIDIA GPU |
| `OMP_NUM_THREADS` | `4` | Число потоков CPU для OpenMP |

## API

### Основные эндпоинты (порт 8090)

```bash
curl http://localhost:8090/health              # статус + сводка
curl http://localhost:8090/stats               # агрегированная статистика
curl http://localhost:8090/objects?limit=20    # список объектов
curl http://localhost:8090/objects/{id}        # детали объекта
curl http://localhost:8090/filters             # доступные фильтры
curl http://localhost:8090/config              # текущая конфигурация (GET/PUT)
curl http://localhost:8090/config/reload       # hot-reload
```

## Скрипты

| Скрипт | Назначение |
|--------|-----------|
| `scripts/deploy.sh` | Развёртывание podman pod |
| `scripts/build-push.sh` | Сборка + push в registry |
| `scripts/test-stream.sh` | Проверка RTSP/ONVIF камер |
| `scripts/health-check.sh` | Мониторинг здоровья |
| `training/train.py` | Дообучение YOLO |
| `training/train.bat` | То же (Windows) |

## Структура проекта

```
cam/
├── config/                # конфигурация
│   ├── cameras.yaml       # RTSP-камеры
│   ├── settings.yaml      # глобальные настройки
│   ├── triggers.yaml      # триггеры
│   └── mqtt/              # конфиг Mosquitto
├── docker/                # Containerfile'ы
├── src/                   # исходный код
│   ├── detection/         # YOLO + CropClassifier
│   ├── recognition/       # LPR, Face, VMR
│   ├── tracking/          # DeepSORT
│   ├── storage/           # PostgreSQL + репозиторий
│   ├── pipeline.py        # основной пайплайн
│   └── main.py            # aiohttp сервер
├── training/              # дообучение
├── ui/                    # веб-интерфейс
├── docker-compose.yml     # Docker Compose
└── .env.example           # шаблон переменных
```
