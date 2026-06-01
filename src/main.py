import asyncio
import json
import os
import signal
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

MSK = timezone(timedelta(hours=3))

def _local(ts):
    """Convert naive UTC datetime to Moscow time isoformat."""
    if ts is None:
        return None
    return ts.replace(tzinfo=timezone.utc).astimezone(MSK).isoformat()

from aiohttp import web
from loguru import logger
from sqlalchemy import select, text

from .config import load_settings, load_cameras, load_triggers
from .pipeline import CameraPipeline
from .stats import StatsCollector
from .storage import init_db, close_db, init_pgvector, init_schema, StorageRepository, get_session
from .storage.models import TrackedObject, FrameCapture
from .actions import ActionDispatcher

# MQTT — optional, enable in settings.yaml
try:
    from .actions import MQTTAction as _MQTTAction
except ImportError:
    _MQTTAction = None
from .training import FineTuner


async def main():
    config_dir = Path(os.environ.get("CONFIG_DIR", "config"))
    settings = load_settings(config_dir)
    cameras_cfg = load_cameras(config_dir)
    triggers_cfg = load_triggers(config_dir)

    models_dir = Path(settings.get("app", {}).get("models_dir", "/app/models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    if "YOLO_CONFIG_DIR" not in os.environ:
        os.environ["YOLO_CONFIG_DIR"] = str(models_dir / "ultralytics")

    log_dir = Path(os.environ.get("LOG_DIR", "/app/logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"cam-analyzer_{datetime.now().strftime('%Y%m%d')}.log"

    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.get("app", {}).get("log_level", "INFO"),
        colorize=True,
    )
    logger.add(
        log_file,
        level=settings.get("app", {}).get("log_level", "INFO"),
        rotation="1 day",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:8} | {name}:{function}:{line} - {message}",
    )

    # --- DB ---
    db_cfg = settings.get("database", {})
    await init_db(
        host=os.environ.get("DB_HOST", db_cfg.get("host", "localhost")),
        port=int(os.environ.get("DB_PORT", db_cfg.get("port", 5432))),
        user=os.environ.get("DB_USER", db_cfg.get("user", "cam")),
        password=os.environ.get("DB_PASSWORD", db_cfg.get("password", "")),
        database=os.environ.get("DB_NAME", db_cfg.get("name", "cam")),
    )
    await init_pgvector()
    await init_schema()
    async with await get_session() as session:
        for stmt in [
            "ALTER TABLE tracked_objects ADD COLUMN IF NOT EXISTS name VARCHAR(128)",
            "ALTER TABLE tracked_objects ADD COLUMN IF NOT EXISTS ignored BOOLEAN DEFAULT FALSE",
        ]:
            try:
                await session.execute(text(stmt))
            except Exception:
                pass
        for idx_stmt in [
            "CREATE INDEX IF NOT EXISTS idx_objects_last_seen ON tracked_objects (last_seen DESC)",
            "CREATE INDEX IF NOT EXISTS idx_objects_ignored_last ON tracked_objects (ignored, last_seen DESC)",
            "CREATE INDEX IF NOT EXISTS idx_objects_name ON tracked_objects (name) WHERE name IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_objects_camera ON tracked_objects (camera_id, last_seen DESC)",
            "CREATE INDEX IF NOT EXISTS idx_objects_class ON tracked_objects (class_name, last_seen DESC)",
            "CREATE INDEX IF NOT EXISTS idx_frames_object ON frame_captures (object_id)",
            "CREATE INDEX IF NOT EXISTS idx_frames_timestamp ON frame_captures (timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_events_object ON events (object_id)",
        ]:
            try:
                await session.execute(text(idx_stmt))
            except Exception:
                pass
        await session.commit()

    data_dir = Path(settings.get("app", {}).get("data_dir", "/data/frames"))
    repository = StorageRepository(data_dir=data_dir)

    # --- MQTT (optional) ---
    if settings.get("mqtt", {}).get("enabled", False) and _MQTTAction is not None:
        mqtt_cfg = settings.get("mqtt", {})
        _MQTTAction.configure(
            host=os.environ.get("MQTT_HOST", mqtt_cfg.get("host", "localhost")),
            port=int(os.environ.get("MQTT_PORT", mqtt_cfg.get("port", 1883))),
            client_id=mqtt_cfg.get("client_id", "cam-analyzer"),
        )
        logger.info("MQTT enabled")

    dispatcher = ActionDispatcher()
    dispatcher.load_triggers(triggers_cfg.get("triggers", []))

    cameras = cameras_cfg.get("cameras", [])
    if not cameras:
        logger.error("No cameras configured")
        return

    # --- Stats ---
    stats_collector = StatsCollector(log_interval=float(settings.get("general", {}).get("stats_log_interval_seconds", 30)))

    # --- Pipelines ---
    pipelines: list[CameraPipeline] = []
    tasks: list[asyncio.Task] = []

    for cam_cfg in cameras:
        pipeline = CameraPipeline(
            camera_config=cam_cfg,
            settings=settings,
            repository=repository,
            dispatcher=dispatcher,
        )
        pipelines.append(pipeline)
        stats_collector._pipelines[pipeline.cam_id] = pipeline.stats
        tasks.append(asyncio.create_task(pipeline.run()))

    logger.info(f"Started {len(tasks)} camera pipelines")

    # --- Stats log background task ---
    async def stats_loop():
        while True:
            await asyncio.sleep(float(settings.get("general", {}).get("stats_log_interval_seconds", 30)))
            stats_collector.maybe_log()

    tasks.append(asyncio.create_task(stats_loop()))

    # --- Periodic cleanup task ---
    async def cleanup_loop():
        from sqlalchemy import delete
        from src.storage.models import FrameCapture, TrackedObject, Event
        while True:
            interval = int(settings.get("general", {}).get("cleanup_interval_seconds", 300))
            await asyncio.sleep(interval)
            try:
                async with await get_session() as session:
                    # Delete frames whose parent object no longer exists
                    alive = select(TrackedObject.id)
                    orph_frames = await session.execute(
                        delete(FrameCapture).where(FrameCapture.object_id.not_in(alive))
                    )
                    # Delete events + objects with zero frames
                    alive_f = select(FrameCapture.object_id).distinct()
                    await session.execute(
                        delete(Event).where(Event.object_id.not_in(alive_f))
                    )
                    orph_objs = await session.execute(
                        delete(TrackedObject).where(
                            ~TrackedObject.id.in_(alive_f)
                        )
                    )
                    await session.commit()
                    if orph_frames.rowcount or orph_objs.rowcount:
                        logger.info(f"Cleanup: removed {orph_frames.rowcount} orphan frames, {orph_objs.rowcount} orphan objects")
            except Exception as ex:
                logger.warning(f"Cleanup error: {ex}")

    tasks.append(asyncio.create_task(cleanup_loop()))

    # --- Fine-tuning (manual only) ---
    fine_tuner = FineTuner(
        models_dir=Path(settings.get("app", {}).get("models_dir", "/app/models")),
        data_dir=data_dir,
        config=settings,
    )
    training_lock = asyncio.Lock()
    training_state: dict[str, dict] = {}

    async def _run_training(name: str | None = None):
        async with training_lock:
            key = name or "__all__"
            training_state[key] = {"status": "collecting", "started_at": _local(datetime.now(timezone.utc))}
            try:
                dataset_dir = await fine_tuner.collect_dataset(name_filter=name)
                if dataset_dir is None:
                    training_state[key] = {"status": "skipped", "reason": "not enough samples", "finished_at": _local(datetime.now(timezone.utc))}
                    return
                training_state[key]["status"] = "training"

                def on_epoch(ep: int, total: int, box: float, cls_loss: float, dfl: float):
                    training_state[key].update({"epoch": ep, "total_epochs": total, "box_loss": round(box, 4), "cls_loss": round(cls_loss, 4)})

                result = await fine_tuner.train(dataset_dir, epoch_callback=on_epoch)
                if result:
                    for pl in pipelines:
                        await pl.reload_classifier(result)
                    training_state[key] = {"status": "done", "model": result, "finished_at": _local(datetime.now(timezone.utc))}
                else:
                    training_state[key] = {"status": "failed", "error": "training returned no model", "finished_at": _local(datetime.now(timezone.utc))}
            except Exception as ex:
                training_state[key] = {"status": "failed", "error": str(ex), "finished_at": _local(datetime.now(timezone.utc))}

    logger.info("Starting HTTP health server...")

    # --- Health HTTP server ---
    health_port = int(os.environ.get("HEALTH_PORT", "8090"))
    health_app = web.Application()

    async def handle_health(request: web.Request) -> web.Response:
        db_ok = False
        try:
            async with await get_session() as session:
                await session.execute(text("SELECT 1"))
            db_ok = True
        except Exception:
            pass

        summary = stats_collector.summary()
        running = summary["cameras_running"]
        if running > 0 and db_ok:
            status = "healthy"
        elif running > 0:
            status = "degraded"
        else:
            status = "unhealthy"

        return web.json_response({"status": status, "db_connected": db_ok, **summary})

    async def handle_stats(request: web.Request) -> web.Response:
        return web.json_response(stats_collector.summary())

    async def handle_detailed(request: web.Request) -> web.Response:
        return web.json_response(stats_collector.snapshot())

    async def handle_live(request: web.Request) -> web.Response:
        frame = await repository.get_latest_frame()
        if frame is None:
            return web.json_response({"error": "no frames"}, status=404)
        if frame.get("timestamp"):
            from datetime import datetime
            ts = datetime.fromisoformat(frame["timestamp"])
            frame["timestamp"] = ts.replace(tzinfo=timezone.utc).astimezone(MSK).isoformat()
        return web.json_response(frame)

    async def handle_list_objects(request: web.Request) -> web.Response:
        camera_id = request.query.get("camera_id")
        class_name = request.query.get("class_name")
        name = request.query.get("name")
        show_ignored = request.query.get("show_ignored") == "1"
        sort = request.query.get("sort", "-last_seen")
        grouped = request.query.get("grouped") == "1"

        if grouped:
            data = await repository.list_grouped(
                camera_id=camera_id, class_name=class_name,
                show_ignored=show_ignored, sort=sort,
            )
            return web.json_response({"total": len(data["groups"]), "groups": data["groups"], "items": data["items"]})

        limit = int(request.query.get("limit", 50))
        offset = int(request.query.get("offset", 0))
        objects = await repository.list_objects(
            camera_id=camera_id, class_name=class_name, name=name, show_ignored=show_ignored,
            limit=limit, offset=offset, sort=sort,
        )
        total = await repository.get_object_count(
            camera_id=camera_id, class_name=class_name, show_ignored=show_ignored,
        )
        items = [
            {
                "id": str(obj.id),
                "camera_id": obj.camera_id,
                "track_id": obj.track_id,
                "class_name": obj.class_name,
                "name": obj.name,
                "ignored": obj.ignored,
                "plate_number": obj.plate_number,
                "face_id": obj.face_id,
                "vmr": (obj.metadata_ or {}).get("vmr"),
                "first_seen": _local(obj.first_seen),
                "last_seen": _local(obj.last_seen),
                "appearance_count": obj.appearance_count,
            }
            for obj in objects
        ]
        return web.json_response({"total": total, "limit": limit, "offset": offset, "items": items})

    async def handle_get_object(request: web.Request) -> web.Response:
        obj_id = uuid.UUID(request.match_info["id"])
        async with await get_session() as session:
            result = await session.execute(select(TrackedObject).where(TrackedObject.id == obj_id))
            obj = result.scalar_one_or_none()
            if obj is None:
                return web.json_response({"error": "not found"}, status=404)
        frames = await repository.list_frames(object_id=obj_id)
        return web.json_response({
            "id": str(obj.id),
            "camera_id": obj.camera_id,
            "track_id": obj.track_id,
            "class_name": obj.class_name,
            "name": obj.name,
            "ignored": obj.ignored,
            "plate_number": obj.plate_number,
            "face_id": obj.face_id,
            "vmr": (obj.metadata_ or {}).get("vmr"),
            "first_seen": _local(obj.first_seen),
            "last_seen": _local(obj.last_seen),
            "appearance_count": obj.appearance_count,
            "frames": [
                {
                    "id": str(f.id),
                    "image": f"/frames/{Path(f.image_path).name}",
                    "bbox": [f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2],
                    "confidence": f.confidence,
                    "timestamp": _local(f.timestamp),
                    "class_name": obj.class_name,
                }
                for f in frames
            ],
        })

    async def handle_patch_object(request: web.Request) -> web.Response:
        obj_id = uuid.UUID(request.match_info["id"])
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        name = body.get("name")
        if name is not None and not isinstance(name, str):
            return web.json_response({"error": "name must be a string"}, status=400)
        if name == "":
            name = None
        ok = await repository.update_object_name(obj_id, name)
        if not ok:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({"status": "updated"})

    async def handle_delete_object(request: web.Request) -> web.Response:
        obj_id = uuid.UUID(request.match_info["id"])
        ok = await repository.delete_object(obj_id)
        if not ok:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({"status": "deleted"})

    async def handle_delete_frame(request: web.Request) -> web.Response:
        frame_id = uuid.UUID(request.match_info["id"])
        ok = await repository.delete_frame(frame_id)
        if not ok:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response({"status": "deleted"})

    async def handle_reclassify_frame(request: web.Request) -> web.Response:
        frame_id = uuid.UUID(request.match_info["id"])
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        new_class = body.get("class_name", "").strip()
        if new_class:
            result = await repository.reclassify_frame(frame_id, new_class)
        else:
            target_name = body.get("name", "").strip()
            if not target_name:
                return web.json_response({"error": "class_name or name required"}, status=400)
            result = await repository.move_frame_to_name(frame_id, target_name)
        if result is None:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(result)

    async def handle_reclassify_group(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        name = body.get("name", "").strip()
        new_class = body.get("class_name", "").strip()
        ids = body.get("ids")
        if not new_class or (not name and not ids):
            return web.json_response({"error": "name or ids + class_name required"}, status=400)
        if ids:
            data = await repository.reclassify_by_ids(ids, new_class)
        else:
            data = await repository.reclassify_group(name, new_class)
        return web.json_response(data)

    async def handle_get_frame(request: web.Request) -> web.Response:
        filename = request.match_info["filename"]
        if ".." in filename or "/" in filename or "\\" in filename:
            return web.json_response({"error": "invalid path"}, status=400)
        frame_dir = Path(settings.get("app", {}).get("data_dir", "/data/frames"))
        filepath = frame_dir / filename
        if not filepath.exists() or not filepath.is_file():
            return web.json_response({"error": "not found"}, status=404)
        return web.FileResponse(filepath)

    async def handle_ui(request: web.Request) -> web.Response:
        ui_file = Path(__file__).resolve().parent.parent / "ui" / "index.html"
        if not ui_file.exists():
            return web.json_response({"error": "UI not found"}, status=404)
        return web.FileResponse(ui_file)

    health_app.router.add_get("/", handle_ui)
    health_app.router.add_get("/ui", handle_ui)
    health_app.router.add_get("/health", handle_health)
    health_app.router.add_get("/stats", handle_stats)
    health_app.router.add_get("/stats/detailed", handle_detailed)
    health_app.router.add_get("/live", handle_live)
    async def handle_list_names(request: web.Request) -> web.Response:
        names = await repository.list_object_names()
        return web.json_response({"names": names})

    async def handle_filters(request: web.Request) -> web.Response:
        names = await repository.list_object_names()
        return web.json_response({
            "cameras": [s.camera_id for s in stats_collector._pipelines.values() if s.running],
            "classes": ["person", "bicycle", "car", "motorcycle", "bus", "truck"],
            "names": [n["name"] for n in names],
        })

    async def handle_settings(request: web.Request) -> web.Response:
        return web.json_response({
            "ui": settings.get("ui", {}),
            "training": {
                "min_samples_per_class": settings.get("training", {}).get("min_samples_per_class", 30),
                "min_show_frames": settings.get("training", {}).get("min_show_frames", 5),
            },
            "detector": {
                "confidence": settings.get("detector", {}).get("confidence", 0.6),
                "model": settings.get("detector", {}).get("model", "yolo11m.pt"),
            },
            "general": {
                "cleanup_interval_seconds": settings.get("general", {}).get("cleanup_interval_seconds", 300),
            },
        })

    async def handle_ignore_object(request):
        obj_id = uuid.UUID(request.match_info["id"])
        changed = False
        async with await get_session() as session:
            result = await session.execute(select(TrackedObject).where(TrackedObject.id == obj_id))
            obj = result.scalar_one_or_none()
            if obj is None:
                return web.json_response({"error": "not found"}, status=404)
            if not obj.ignored:
                obj.ignored = True
                changed = True
                await session.commit()
        # Also mark the in-memory active track as ignored
        if changed:
            for pl in pipelines:
                pl.mark_track_ignored(camera_id=obj.camera_id, track_id=obj.track_id)
        return web.json_response({"ok": True, "ignored": True})

    async def handle_training_status(request: web.Request) -> web.Response:
        total = await fine_tuner.count_labeled_samples()
        return web.json_response({
            "enabled": fine_tuner.enabled,
            "labeled_samples": total,
            "min_samples_per_class": fine_tuner.min_samples,
            "running": training_state,
        })

    async def handle_training_candidates(request: web.Request) -> web.Response:
        candidates = await fine_tuner.list_candidates()
        return web.json_response({"candidates": candidates, "min_samples": fine_tuner.min_samples, "running": training_state})

    async def handle_training_export(request: web.Request) -> web.Response:
        name = request.query.get("name") or None
        zip_path = await fine_tuner.export_zip(name_filter=name)
        if zip_path is None:
            return web.json_response({"error": "not enough samples or no named objects"}, status=404)
        filename = zip_path.name
        return web.FileResponse(zip_path, headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "application/zip",
        })

    async def handle_training_trigger(request: web.Request) -> web.Response:
        name = request.query.get("name") or None
        key = name or "__all__"
        if training_state.get(key, {}).get("status") in ("collecting", "training"):
            return web.json_response({"status": "error", "message": "already running"}, status=409)
        asyncio.create_task(_run_training(name))
        return web.json_response({"status": "started", "name": name or "all"})

    async def handle_config_reload(request: web.Request) -> web.Response:
        nonlocal settings, cameras_cfg, triggers_cfg
        logger.info("Reloading configuration...")
        ...

    async def handle_model_reset(request: web.Request) -> web.Response:
        import os
        fine_tuned = Path(settings.get("app", {}).get("models_dir", "/app/models")) / "fine-tuned.pt"
        if not fine_tuned.exists():
            return web.json_response({"status": "ok", "message": "No fine-tuned model found"})
        fine_tuned.unlink()
        logger.info("Fine-tuned model deleted, disabling classifier")
        for pl in pipelines:
            await pl.reload_classifier(str(fine_tuned))
        return web.json_response({"status": "ok", "message": "Classifier disabled"})

        # Update triggers
        dispatcher.load_triggers(new_triggers_cfg.get("triggers", []))
        logger.info("Triggers reloaded")

        # Update existing pipelines with new detector/tracker settings
        new_cam_map = {c["id"]: c for c in new_cameras_cfg.get("cameras", [])}
        old_ids = {pl.cam_id for pl in pipelines}
        new_ids = set(new_cam_map.keys())

        # Stop removed cameras
        for pl in list(pipelines):
            if pl.cam_id not in new_ids:
                logger.info(f"Stopping pipeline for removed camera: {pl.cam_id}")
                pl.stop()
                pipelines.remove(pl)
                stats_collector._pipelines.pop(pl.cam_id, None)

        # Update existing or reconfigure
        for pl in pipelines:
            if pl.cam_id in new_cam_map:
                await pl.reconfigure(new_cam_map[pl.cam_id], new_settings)
                logger.info(f"Pipeline {pl.cam_id} reconfigured")

        # Add new cameras
        for cam_id in new_ids - old_ids:
            cam_cfg = new_cam_map[cam_id]
            pipeline = CameraPipeline(
                camera_config=cam_cfg,
                settings=new_settings,
                repository=repository,
                dispatcher=dispatcher,
            )
            pipelines.append(pipeline)
            stats_collector._pipelines[pipeline.cam_id] = pipeline.stats
            asyncio.create_task(pipeline.run())
            logger.info(f"Added pipeline for new camera: {cam_id}")

        settings = new_settings
        cameras_cfg = new_cameras_cfg
        triggers_cfg = new_triggers_cfg
        return web.json_response({
            "status": "ok",
            "cameras": len(pipelines),
            "cameras_added": list(new_ids - old_ids),
            "cameras_removed": list(old_ids - new_ids),
        })

    health_app.router.add_get("/training/status", handle_training_status)
    health_app.router.add_get("/training/candidates", handle_training_candidates)
    health_app.router.add_get("/training/export", handle_training_export)
    health_app.router.add_post("/training/run", handle_training_trigger)
    health_app.router.add_post("/config/reload", handle_config_reload)
    health_app.router.add_post("/model/reset", handle_model_reset)

    health_app.router.add_get("/objects", handle_list_objects)
    health_app.router.add_get("/objects/names", handle_list_names)
    health_app.router.add_get("/filters", handle_filters)
    health_app.router.add_get("/settings", handle_settings)
    health_app.router.add_get("/objects/{id}", handle_get_object)
    health_app.router.add_patch("/objects/{id}", handle_patch_object)
    health_app.router.add_delete("/objects/{id}", handle_delete_object)
    health_app.router.add_post("/objects/{id}/ignore", handle_ignore_object)
    health_app.router.add_delete("/frames/{id}", handle_delete_frame)
    health_app.router.add_post("/frames/{id}/reclassify", handle_reclassify_frame)
    health_app.router.add_post("/objects/reclassify-group", handle_reclassify_group)
    health_app.router.add_get("/frames/{filename:.+}", handle_get_frame)

    runner = web.AppRunner(health_app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", health_port)
    await site.start()
    logger.info(f"Health endpoint: http://0.0.0.0:{health_port}/health")

    # --- Shutdown ---
    loop = asyncio.get_running_loop()

    def shutdown():
        logger.info("Shutting down...")
        for pl in pipelines:
            pl.stop()
        for t in tasks:
            t.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
            pass

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()
        if _MQTTAction is not None:
            await _MQTTAction.disconnect()
        await close_db()
        logger.info("Shutdown complete")


def run():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
