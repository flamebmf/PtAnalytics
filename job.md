# cam — Video Analytics System

## Status: Deployed on Debian (bookworm), 3 cameras running, collecting data

### Today (2026-05-29) — Session 2
- [x] **Fix: `delete_frame` updates parent `last_seen`** — after frame deletion, recalculates `TrackedObject.last_seen` from remaining frames' max timestamp
- [x] **Fix: `renderGrouped` respects sort direction** — group+unnamed items now sort by the selected `sort` param (not always DESC)
- [x] **Fix: timezone error in `delete_frame`** — `func.max()` returns tz-aware datetime, but column is `TIMESTAMP WITHOUT TIME ZONE`. `_db_timestamp()` strips tzinfo after UTC conversion. Error was: `can't subtract offset-naive and offset-aware datetimes` (asyncpg `DataError`)
- [x] **Script: `cleanup-small-frames.py`** — deletes frames with bbox w < 40 or h < 40, removes image files, recalculates parent `last_seen`, reports orphans
- [x] **Periodic cleanup task** — runs every 5 min: deletes orphan frames (no parent object) + orphan objects (0 frames). Logs only when something removed

### Session 1 (2026-05-29)
- [x] Fix: class-aware tracker matching — prevents cat+person or car+car merging
- [x] Fix: local model resolution (`_find_model`) — searches CWD, ultralytics cache, YOLO_CONFIG_DIR before YOLO() download attempt
- [x] Fix: validate OpenVINO model integrity — re-exports if .bin is corrupt
- [x] Fix: explicit `task=detect` for OpenVINO export
- [x] OpenVINO backend added (torch/openvino per `detector.backend`), with per-camera override
- [x] Migrate: UBI9 → Debian bookworm (default Containerfile.debian, `--ubi9` for UBI)
- [x] Fix: `motion_skip_seconds: 1.0` removed from cameras.yaml (was overriding global `skip_seconds: 0.0`)
- [x] Fix: gate1 `motion_enabled` was `true` — set to `false`
- [x] Full frame save with bbox overlay instead of cropped object
- [x] `skip_seconds: 0.0` (global), `workers: 4`, `backend: torch`
- [x] Timing instrumentation: `[TIMING cam_name] read() took Xs` for read() latency
- [x] Person-in-car filter: removes person detections with >50% overlap inside vehicle bboxes
- [x] Per-camera confidence: gate1=0.4, parking1=0.6, entrance=0.6
- [x] Vehicle ReID: `src/recognition/reid.py` — 512-dim HSV+shape embedding, pgvector cosine search
- [x] ReID wired into pipeline: auto-embeds unnamed vehicles on first frame, searches other cameras, auto-assigns name on match >0.85
- [x] Repository: `find_similar_objects()` (cosine distance via pgvector), `update_embedding()`
- [x] Git tag: `v2026-05-29-before-reid` for rollback
- [x] Commit: `be2f9a0` — person-in-car filter + per-camera confidence + vehicle ReID
- [x] Dataset collection: 3 crops/track (entry/mid/exit) + dedup (IoU>0.5 within 60s)
- [x] CropSample model + `save_crop()`: clean JPG + YOLO .txt label, stored in `/data/crops/{class}/{camera}/`
- [x] `export-dataset.py`: splits crops → train/val + generates dataset.yaml
- [x] ReID fix: `find_similar_objects()` now uses offset-naive UTC for DB query

## Known Issues
- OpenVINO on yolo11s yields 1.4 FPS / 98% CPU (worse than torch's 0.27s). Works on Debian, no gain on small models.
- False detections: glare detected as cars/people, people as cars (`confidence: 0.55`). Mitigated by person-in-car filter + per-camera thresholds.
- Wide camera removed (unstable RTSP, detections not useful).
- **LPR**: YOLO plate model не загружается — нет логов ни о загрузке, ни об ошибке (вероятно тихо падает в `_find_model` или при даунлоаде)
- **entrance**: 0 детекций с момента деплоя (confidence 0.6 слишком высок для этого ракурса)
- **parking1**: всего 1 объект (стоящие машины не проходят motion filter)

## Config
```yaml
detector:
  model: yolo11s.pt
  confidence: 0.55          # global default; overridden per-camera
  imgsz: 1280
  workers: 4
  backend: torch
motion:
  skip_seconds: 0.0
cameras:
  gate1:  { confidence: 0.4 }   # lower threshold for passing cars at gate
  parking1: { confidence: 0.6 }  # higher threshold suppresses stationary false positives
  entrance: { confidence: 0.6 }
```

## Recent Commits
| Hash | Message |
|------|---------|
| `80ec357` | fix: delete_frame converts max_ts via _db_timestamp() |
| `121248f` | feat: cleanup-small-frames.py script |
| `efec9bd` | fix(ui): renderGrouped respects selected sort direction |
| `abc9d90` | fix: delete_frame updates parent last_seen to max remaining frame timestamp |

## Scripts
| Script | Purpose |
|--------|---------|
| `scripts/backfill-reid.py` | Computes embeddings for existing unnamed vehicles, searches matches across cameras |
| `scripts/export-dataset.py` | Splits collected crops into train/val, generates dataset.yaml for fine-tuning |
| `scripts/cleanup-small-frames.py` | Removes frames with bbox < min_bbox_size, cleans up orphaned objects |

## API
| Endpoint | Status |
|----------|--------|
| `GET /objects` | Working — filter by camera_id, class_name, name, show_ignored, limit, offset |
| `GET /objects/names` | Working — distinct names with count + cameras |
| `GET /objects/{id}` | Working — detail + frames |
| `PATCH /objects/{id}` | Working — rename |
| `DELETE /objects/{id}` | Working — cascade delete |
| `POST /objects/{id}/ignore` | Working — toggle ignore |
| `GET /frames/{filename}` | Working — serve JPG |
| `GET /health` | Working |
| `GET /stats` | Working |

## Next Steps
- [ ] **Deploy dataset collection**, let it run for 3-7 days
- [ ] **Run `export-dataset.py`**, fine-tune YOLO on collected crops
- [ ] **Fix LPR** — YOLO plate model не скачивается или PaddleOCR не читает номера
- [ ] **Unlink API** (`PATCH /objects/{id}/unlink`): reset auto-assigned name to NULL
- [ ] **Manual merge** (`POST /objects/merge`): assign same name to two+ TrackedObjects
- [ ] **Pending merge review API**: show match candidates awaiting confirmation
- [ ] **Multi-frame confirmation**: only auto-assign name after match persists for N consecutive frames
- [ ] **Tune ReID threshold** (currently 0.85) based on real backfill scores
- [ ] **PresenceTracker** — arriving/home/leaving по именам объектов (gate1→parking1→entrance)

## Done
- [x] Backfill script run: 26 vehicles processed, all have embeddings
- [x] ReID pipeline active: auto-computes embedding on new tracks, cross-camera matching
- [x] min_bbox_size=40: filters glares and tiny false detections
