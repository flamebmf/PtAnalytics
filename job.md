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

## Known Issues
- OpenVINO on yolo11s yields 1.4 FPS / 98% CPU (worse than torch's 0.27s). Works on Debian, no gain on small models.
- False detections: glare detected as cars/people, people as cars (`confidence: 0.55`).
- Wide camera removed (unstable RTSP, detections not useful).

## Config
```yaml
detector:
  model: yolo11s.pt
  confidence: 0.55
  imgsz: 1280
  workers: 4
  backend: torch                # torch, openvino
motion:
  skip_seconds: 0.0
```

## API
| Endpoint | Status |
|----------|--------|
| `GET /objects` | Working — camera_id, class_name, name, show_ignored, limit, offset |
| `GET /objects/names` | Working — distinct names with count+cameras |
| `GET /objects/{id}` | Working — detail + frames |
| `PATCH /objects/{id}` | Working — rename |
| `DELETE /objects/{id}` | Working — cascade delete |
| `POST /objects/{id}/ignore` | Working — toggle ignore |
| `GET /frames/{filename}` | Working — serve JPG |
| `GET /health` | Working |
| `GET /stats` | Working |

## Next Steps
- [ ] Tune `confidence: 0.7` to reduce false detections, re-evaluate
- [ ] PresenceTracker — система "подъезжает/дома/уехал" по именам объектов
- [ ] Object profiles — vehicle ReID (ONNX) для переидентификации машин без номера
- [ ] Fine-tune YOLO на своих камерах (если 0.7 отсекает слишком много)
