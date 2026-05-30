# cam — Video Analytics System

## Status: Deployed on Debian (bookworm), 3 cameras running, auto fine-tuning active

### Today (2026-05-30) — Session 3
- [x] **Pipeline fps throttle** — `_frame_interval=1/fps` + `asyncio.sleep()` ensures processing respects camera `fps`
- [x] **Tracker class voting** — `KalmanBoxTracker._class_votes` tracks class distribution; switches only when majority wins by ≥2 votes (fixes person↔car flips)
- [x] **Model yolo11s.pt → yolo11m.pt**, confidence `0.55` → `0.6`, removed per-camera confidence overrides
- [x] **Periodic fine-tuning** — `src/training/fine_tuner.py`: collects crops from user-named objects grouped by class, exports YOLO dataset, trains via `model.train()`, hot-swaps model
- [x] **Training API**: `GET /training/status` (stats), `POST /training/run` (trigger manually)
- [x] **Model persistence** — `YOLO_CONFIG_DIR` set to `/app/models/` (volume), never re-downloads after rebuild
- [x] **UI: `/filters` endpoint** — lightweight camera/class/name lists (replaced `/objects?limit=500`)
- [x] **UI: name autocomplete** — `<datalist>` on filter-name + detail-name inputs, suggests existing names
- [x] **UI: SQL GROUP BY for grouped view** — `/objects?grouped=1` backend GROUP BY instead of JS iterating 5000 objects
- [x] **`POST /config/reload`** — hot-reloads cameras.yaml, settings.yaml, triggers.yaml without restart
- [x] **`pipeline.reconfigure()`** — updates detector (confidence, imgsz, classes), tracker params, motion, fps on the fly
- [x] **`pipeline.reload_detector()`** — hot-swaps YOLO model after fine-tuning
- [x] **UI: `↻ Config` button** — one-click config reload with visual confirmation

### Session 2 (2026-05-29)
- [x] Fix: `delete_frame` updates parent `last_seen` to max remaining frame timestamp
- [x] Fix: `renderGrouped` respects selected sort direction
- [x] Fix: timezone error in `delete_frame` — `_db_timestamp()` strips tzinfo
- [x] Script: `cleanup-small-frames.py` — deletes frames with small bbox
- [x] Periodic cleanup task — every 5 min: deletes orphan frames + objects
- [x] Fix: `last_seen` uses frame timestamp via `get_or_create_object(timestamp=...)`
- [x] Script: `backfill-last-seen.py` — recalculates `first_seen`/`last_seen` from frame timestamps
- [x] Fix: removed `onupdate=func.now()` from `last_seen` column
- [x] Fix: `_unlocal` in backfill — UTC→MSK conversion mismatch
- [x] Fix: cleanup FK violation — delete events before orphan objects
- [x] Fix: disable heuristic plate fallback — Canny+contour found wheel disks as plates

### Session 1 (2026-05-29)
- [x] Class-aware tracker matching — per-class IoU prevents cross-class merges
- [x] Local model resolution — `_find_model` searches multiple paths before download
- [x] OpenVINO model integrity — re-exports corrupt .bin
- [x] OpenVINO backend support (torch/openvino)
- [x] Debian bookworm migration (Containerfile.debian)
- [x] Full frame save with bbox overlay
- [x] Person-in-car filter — removes persons >50% inside vehicle bbox
- [x] Vehicle ReID — 512-dim HSV+shape embedding, pgvector cosine search
- [x] Dataset collection — 3 crops/track (entry/mid/exit) with dedup
- [x] `export-dataset.py` — train/val split + dataset.yaml

## Config
```yaml
detector:
  model: yolo11m.pt          # was yolo11s.pt
  confidence: 0.6            # was 0.55
  imgsz: 1280
  workers: 4
  backend: torch
  classes: [0,1,2,3,5,7]     # person, bicycle, car, motorcycle, bus, truck
  min_bbox_size: 40
tracker:
  depart_timeout: 10.0
motion:
  skip_seconds: 0.0
training:
  enabled: true
  check_interval_hours: 6
  min_samples_per_class: 30
  epochs: 30
  imgsz: 1280
  base_model: yolo11m.pt
```

## API
| Endpoint | Description |
|----------|-------------|
| `GET /health` | Status + DB check |
| `GET /stats`, `/stats/detailed` | Per-camera metrics |
| `GET /filters` | Cameras, classes, names for UI dropdowns |
| `GET /objects` | List — filter by camera_id, class_name, name, sort; `?grouped=1` for GROUP BY |
| `GET /objects/names` | Distinct names with count + cameras |
| `GET /objects/{id}` | Detail + frames |
| `PATCH /objects/{id}` | Rename object |
| `DELETE /objects/{id}` | Cascade delete |
| `POST /objects/{id}/ignore` | Toggle ignore |
| `DELETE /frames/{id}` | Delete single frame, recalc parent last_seen |
| `GET /frames/{filename}` | Serve JPG |
| `POST /config/reload` | Hot-reload cameras.yaml, settings.yaml, triggers.yaml |
| `GET /training/status` | Labeled samples count, last train time |
| `POST /training/run` | Trigger fine-tuning manually |

## Scripts
| Script | Purpose |
|--------|---------|
| `scripts/deploy.sh` | Podman pod deploy (PG + MQTT optional) |
| `scripts/build-push.sh` | Build image + push to registry |
| `scripts/test-stream.sh` | RTSP/ONVIF camera verification |
| `scripts/health-check.sh` | Pod/container/DB health check |
| `scripts/backfill-reid.py` | Compute embeddings for existing vehicles |
| `scripts/backfill-last-seen.py` | Recalculate first_seen/last_seen from frame timestamps |
| `scripts/cleanup-small-frames.py` | Remove frames with small bbox |
| `scripts/export-dataset.py` | Split crops → train/val for fine-tuning |

## Architecture
```
RTSP → StreamReader → MotionDetector → YoloDetector → DeepSortTracker → LPR/Face/ReID → Storage(PG+pgvector) → Actions(webhook/MQTT/log)
                                                                              ↓
                                                                       FineTuner (auto train on named objects)
```

## Known Issues
- OpenVINO on small models: 1.4 FPS / 98% CPU (worse than torch)
- **LPR**: YOLO plate model не загружается
- **entrance**: 0 детекций с confidence 0.6 (было) — должно улучшиться с yolo11m + 0.6
- **parking1**: мало объектов (стоящие машины не проходят motion filter)

## Next Steps
- [ ] **Fix LPR** — YOLO plate model download + PaddleOCR integration
- [ ] **Multi-frame confirmation** — assign name only after match persists for N frames
- [ ] **PresenceTracker** — arriving/home/leaving по именам объектов
- [ ] **Manual merge** (`POST /objects/merge`) — link two objects under same name
- [ ] **Tune ReID threshold** based on backfill scores
- [ ] **Run dataset collection for 3-7 days**, then verify fine-tuning quality
