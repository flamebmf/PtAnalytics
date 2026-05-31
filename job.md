# cam — Video Analytics System

## Status: Deployed on Debian (bookworm), 2 cameras (parking1, entrance), manual fine-tuning active

### Today (2026-05-30) — Session 3
- [x] **Pipeline fps throttle** — `_frame_interval=1/fps` + `asyncio.sleep()`
- [x] **Tracker class voting** — switches class only when majority wins by ≥2 votes
- [x] **Model yolo11s.pt → yolo11m.pt**, confidence `0.55` → `0.6`
- [x] **Per-frame reclassify** — dropdown on each frame to change class, moves frame to new object preserving name
- [x] **Per-frame move to group** — input with autocomplete, moves frame to named group
- [x] **Group class** — button in group detail to set class for all objects with same name
- [x] **DB indexes** — 8 indexes added for query performance (last_seen, ignored, name, camera, class, frames, events)
- [x] **get_or_create_object** — `ORDER BY last_seen DESC LIMIT 1` handles duplicates from reclassification
- [x] **Manual fine-tuning** — `src/training/fine_tuner.py`: per-name training, stdout capture, clean epoch logging
- [x] **Training by name** — `collect_dataset(name_filter=...)` merges all classes under dominant class
- [x] **Training UI:** modal with per-name candidates, run/run-all buttons, live progress polling
- [x] **Model persistence** — `fine-tuned.pt` auto-loaded on startup if present
- [x] **All timestamps MSK (+3)** — `/live`, objects, frames
- [x] **`POST /config/reload`** — hot-reloads cameras/settings/triggers, adds/removes pipelines
- [x] **`pipeline.reconfigure()`** — updates detector/tracker/motion/fps on the fly
- [x] **`pipeline.reload_detector()`** — hot-swaps YOLO model after fine-tuning
- [x] **UI redesign:** PlurumTech brand — dark theme #04070d, Roboto font, gradient Pt logo, neon buttons
- [x] **Live preview** — 25vw image with class badge overlay + structured text info
- [x] **Merged navbar:** logo + filters + buttons + health in one row
- [x] **Removed gate1** from cameras.yaml

### Session 2 (2026-05-29)
- [x] Fix: `delete_frame` updates parent `last_seen`
- [x] Fix: `renderGrouped` respects sort direction
- [x] Fix: timezone errors in `delete_frame`, `_db_timestamp()` strips tzinfo
- [x] Script: `cleanup-small-frames.py`
- [x] Periodic cleanup — every 5 min: orphan frames + objects + events
- [x] Fix: `last_seen` uses frame timestamp via `get_or_create_object(timestamp=...)`
- [x] Script: `backfill-last-seen.py`
- [x] Fix: removed `onupdate=func.now()` from `last_seen` column
- [x] Fix: `_unlocal` in backfill
- [x] Fix: cleanup FK violation
- [x] Fix: disable heuristic plate fallback

### Session 1 (2026-05-29)
- [x] Class-aware tracker matching, local model resolution, OpenVINO support
- [x] Debian bookworm, full frame save, person-in-car filter
- [x] Vehicle ReID, dataset collection, export-dataset.py

## Config
```yaml
detector:
  model: yolo11m.pt
  confidence: 0.6
  imgsz: 1280, workers: 4, backend: torch
  classes: [person, bicycle, car, motorcycle, bus, truck]
  min_bbox_size: 40
tracker:
  depart_timeout: 10.0, max_age: 30
motion:
  skip_seconds: 0.0
training:
  min_samples_per_class: 30, epochs: 30, imgsz: 1280
cameras:
  parking1: { confidence: 0.6, imgsz: 1280, fps: 10 }
  entrance: { confidence: 0.6, imgsz: 1280, fps: 10 }
```

## API
| Endpoint | Description |
|----------|-------------|
| `GET /health`, `/stats`, `/stats/detailed` | Health + per-camera metrics |
| `GET /filters` | Cameras, classes, names |
| `GET /live` | Latest frame with object info |
| `GET /objects` | List with filters; `?grouped=1` for SQL GROUP BY |
| `GET /objects/names` | Distinct names |
| `GET /objects/{id}` | Detail + frames (with class_name per frame) |
| `PATCH /objects/{id}` | Rename |
| `DELETE /objects/{id}` | Cascade delete |
| `POST /objects/{id}/ignore` | Toggle ignore |
| `POST /objects/reclassify-group` | Set class for all objects with same name |
| `DELETE /frames/{id}` | Delete frame, recalc parent |
| `POST /frames/{id}/reclassify` | Change class or move to group |
| `GET /frames/{filename}` | Serve JPG |
| `POST /config/reload` | Hot-reload configs |
| `GET /training/candidates` | Per-name candidates with frame counts |
| `GET /training/status` | Running jobs state |
| `POST /training/run?name=X` | Trigger fine-tuning |

## Architecture
```
RTSP → StreamReader → MotionDetector → YoloDetector → DeepSortTracker → LPR/Face/ReID → Storage(PG+pgvector) → Actions
                                                                              ↓
                                                                       FineTuner (manual, per-name)
```

## Known Issues
- **LPR**: YOLO plate model не загружается
- **parking1**: мало объектов (стоящие машины не проходят motion filter)

## Next Steps
- [ ] **Fix LPR** — YOLO plate model download + PaddleOCR integration
- [ ] **PresenceTracker** — arriving/home/leaving по именам объектов
- [ ] **Tune ReID threshold** based on backfill scores
