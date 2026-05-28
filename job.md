# cam — Video Analytics System

## Status: Running on RHEL — capture, detection, events working

### Last analysis (2026-05-28): 11 issues found, 7 fixed
### Code review (2026-05-28): All fixes confirmed in source. 3 minor issues remain. 1 new issue found (see below).

### Latest camera pipeline fix (2026-05-28)
- [x] Root cause identified: long-lived OpenCV `VideoCapture.read()` blocked after first frame on RTSP.
- [x] Track lifecycle management: frames saved + triggers fired only on APPEARED/DEPARTED events, not on every frame for unchanged tracks.
- [x] `StreamReader` now uses persistent RTSP capture with timeout/reconnect protection.
- [x] `/stats` and STATS logs now expose captured/skipped/motion/processed/detections per camera.
- [x] `python -m compileall -q src run.py` passes locally.
- [x] Verified on RHEL: 2/2 cameras running, frames captured, YOLO detections, `any_car` events fired.
- [x] Fixed DB timestamp timezone errors for object/frame persistence.
- [x] Fixed PaddleOCR init argument compatibility (`show_log` removed).
- [x] Fixed DeepSort Kalman bbox shape error.
- [x] Added model/cache mounts via `/srv/cam-analyzer/models` (`YOLO_CONFIG_DIR`, `PADDLE_HOME`, `PADDLEX_HOME`, `HOME`, cache).
- [x] Added `debug` option to control verbose capture logs.
- [x] Added YOLO CPU tuning: `detector.imgsz`, `detector.workers`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`.

**Critical (fixed):**
- [x] Missing `import asyncio` in stream_reader.py:63
- [x] `.tolist()` on already-list in pipeline.py:182
- [x] Sync MQTT `connect()` blocking event loop — wrapped in `loop.run_in_executor()`
- [x] Alembic async URL with sync engine — changed to `postgresql://`

**Moderate (fixed):**
- [x] Import ordering in db.py — `text` moved to top
- [x] Deprecated `datetime.utcnow()` → `datetime.now(timezone.utc)` in repository.py

**Minor (not urgent):**
- [ ] Private `_pipelines` access in main.py:89 — should use `StatsCollector.register()`
- [ ] `podman exec` into pod in health-check.sh:78 — should use container name
- [ ] Regex YAML parsing in test-stream.sh — fragile, consider PyYAML
- [x] `.env.example` had duplicate `WITH_LOCAL_PG=no` (lines 19, 39) — removed line 39

## Completed
- [x] Project structure + configs (YAML)
- [x] RTSP stream capture with auto-reconnect (OpenCV persistent capture + timeout/reconnect)
- [x] Motion detection (MOG2 / frame diff)
- [x] YOLO object detection + class filtering
- [x] DeepSORT multi-object tracking (Kalman + IoU)
- [x] PostgreSQL + pgvector models (SQLAlchemy async)
- [x] Storage repository (upsert objects, save frames, log events)
- [x] License plate recognition (PaddleOCR)
- [x] Face detection + recognition (InsightFace ArcFace)
- [x] Action system: webhook, MQTT, log dispatcher
- [x] CameraPipeline orchestrator
- [x] main.py — multi-camera async runner
- [x] Containerfile (UBI 9 main, Debian fallback) — Added /app/logs, /app/config dirs
- [x] podman-compose.yml + Mosquitto service
- [x] scripts/deploy.sh — podman pod + local PG/MQTT
- [x] scripts/deploy-pg.sh — PG+pgvector standalone
- [x] scripts/install-deps.sh — RHEL 9/10 deps
- [x] scripts/install-systemd.sh — systemd autostart
- [x] scripts/test-stream.sh — Updated with camera count, fixed parser, added health_cmd after chown
- [x] Stats collector (PipelineStats, StatsCollector)
- [x] HTTP health endpoint (:8090/health, /stats, /stats/detailed)
- [x] scripts/health-check.sh — check pod/containers/health/DB
- [x] Health monitor for PostgreSQL, MQTT broker, cam-analyzer containers
- [x] Volume mounts: config (rw), logs, frames, models, pgdata
- [x] Model cache mount: `/srv/cam-analyzer/models` → `/app/models`
- [x] Startup snapshots saved to `/srv/cam-analyzer/frames/`
- [x] Per-trigger once-per-object deduplication (`once_per_object`)
- [x] Debug logging controls (`app.debug`, camera `debug`)
- [x] CPU inference tuning (`detector.workers`, `detector.imgsz`)
- [x] OpenVINO backend (torch/openvino) — `detector.backend` в settings.yaml, per-camera override через `cameras.yaml → detector.backend`, одноразовый экспорт .pt → OpenVINO IR, requirements.txt: `openvino-dev>=2024.0.0`

## Observability

### Логи
- **Local path**: `/srv/cam-analyzer/logs/cam-analyzer_YYYYMMDD.log`
- **Container path**: `/app/logs/cam-analyzer_YYYYMMDD.log`
- Каждые 30 сек в лог пишется сводка STATS (кадры/объекты/события/ошибки)

### HTTP эндпоинты (порт 8090)
| Endpoint | Ответ | Status |
|----------|-------|--------|
| `/health` | `{"status": "healthy", ...}` | **Working** |
| `/stats` | Сводка по всем камерам | **Working** |
| `/stats/detailed` | Детальные счётчики на каждую камеру | **Working** |
| `/` или `/ui` | Web UI для просмотра объектов и изображений | **Working** |
| `/objects` | Список обнаруженных объектов (camera_id, class_name, limit, offset) | **Working** |
| `/objects/{id}` | Детали объекта + список его frame-изображений | **Working** |
| `/frames/{filename}` | JPG-файл frame-изображения | **Working** |

### Snapshot кадров
После запуска/рестарта каждой камеры автоматически сохраняется один кадр в `/data/frames/snapshot_{camera_id}_{timestamp}.jpg`
- Локальный путь: `/srv/cam-analyzer/frames/snapshot_*.jpg`

### Performance
- Capture FPS now reported separately as `capture_fps` in `/stats` and logs.
- Processing FPS (`fps`) is YOLO/tracking throughput, not raw camera capture rate.
- CPU tuning defaults: `detector.workers: 4`, `detector.imgsz: 1280`, `backend: openvino`.
- bottleneck was per-camera `motion_skip_seconds: 1.0` in cameras.yaml overriding global `skip_seconds`. Removed from cameras.yaml.
- FPS: 0.7->2.2 per camera after removing skip_seconds override. CPU at ~50% with workers=4.
- OpenVINO backend (~3x YOLO speedup on Intel CPU, ~15-20% CPU). Export happens once per model version.

### Health check
```bash
bash scripts/health-check.sh          # разово
bash scripts/health-check.sh --watch  # мониторинг каждые 10с
```

## Next steps
- [x] Fix Docker ffmpeg image tag (jrottenberg/ffmpeg:7.1 → 7.1-ubuntu)
- [x] Add health_monitor probes to MQTT broker and cam-analyzer containers
- [x] Mount config directory as read-write (`:Z`) for live updates
- [x] Add log file output (/app/logs/*.log)
- [x] Rebuild on RHEL and verify `captured` grows in `/stats`
- [x] Verify detections/events on both cameras
- [x] Track lifecycle management: save frames + fire triggers only on APPEARED and DEPARTED, skip unchanged tracks
- [x] Web UI: список объектов с фильтрами (камера, класс, имя), просмотр frame, редактирование имени, удаление объекта
- [x] API: PATCH /objects/{id} (изменить имя), DELETE /objects/{id} (удалить объект + кадры + события)
- [x] Per-camera detector overrides (imgsz, confidence) в cameras.yaml под `detector:`
- [x] OPENCV_FFMPEG_READ_ATTEMPTS увеличен с 128 до 1024 для камер с аудиопотоками
- [x] Merge by name: объекты с одинаковым именем считаются одним (триггер 1 раз на имя, группировка в UI)
- [x] Centroid-distance fallback в трекере — матчинг быстрых объектов при низком FPS (порог `(w+h)/2 * 3.0`)
- [x] Per-object ignore flag: колонка `ignored` в tracked_objects, API POST /objects/{id}/ignore, pipeline пропускает кадры+триггеры для ignored-объектов, кнопка "Игнор" в UI
- [x] imgsz=1280, workers=16 для улучшения детекции на обзорных камерах
- [x] Consider async inference worker pool for better use of 72 CPU cores — resolved via OpenVINO backend, ~2-3x faster per-inference
- [ ] `bash scripts/install-deps.sh` на RHEL хосте
- [x] `bash scripts/deploy.sh` — OK (with systemd fix)
- [ ] GPU ускорение — тестирование
- [ ] Web UI (FastAPI) — будущая фаза
- [ ] Prometheus /metrics endpoint
