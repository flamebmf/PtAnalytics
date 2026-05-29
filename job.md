# cam — Video Analytics System

## Status: Deployed on Debian (bookworm), 3 cameras running, collecting data

### Today (2026-05-29)
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
- [ ] **Fix LPR** — ни разу не определился номер на машине
- [ ] **Backfill script** (`scripts/backfill-reid.py`): iterate unnamed vehicles with saved frames, compute embedding, store, cluster by similarity, auto-assign names
- [ ] **Unlink API** (`PATCH /objects/{id}/unlink`): reset auto-assigned name to NULL, log reason
- [ ] **Manual merge** (`POST /objects/merge`): assign same name to two+ TrackedObjects
- [ ] **Pending merge review** (`GET /objects/pending-merge`): show match candidates awaiting confirmation
- [ ] **Multi-frame confirmation**: only auto-assign name after match persists for N consecutive frames
- [ ] **Tune ReID threshold** (currently 0.85) based on real backfill data
- [ ] **Deploy with ReID**, collect cross-camera match data
- [ ] **PresenceTracker** — arriving/home/leaving по именам объектов (gate1→parking1→entrance)
