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

### Файлы

Все конфиги в `config/`:
- `cameras.yaml` — RTSP-камеры (копия из .example, вписать свои rtsp_url)
- `settings.yaml` — пороги детекции, motion, трекер, LPR, face, VMR
- `triggers.yaml` — триггеры и действия (webhook/mqtt/log)

### UI Settings

Большинство настроек доступно через веб-интерфейс (шестерёнка в правом верхнем углу).
Изменения применяются через `PUT /config` и hot-reload пайплайнов без перезапуска.

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

Процесс:

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
